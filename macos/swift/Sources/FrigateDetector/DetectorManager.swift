import Foundation
import AppKit

enum DetectorStatus: String {
    case stopped = "Stopped"
    case starting = "Starting..."
    case running = "Running"
    case stopping = "Stopping..."
    case restarting = "Restarting..."
}

class DetectorManager: ObservableObject {
    @Published var status: DetectorStatus = .stopped
    @Published var endpoints: [String] = []
    @Published var startupEnabled = false
    @Published var models: [(name: String, sizeMB: Double)] = []
    @Published var cliInstalled = false
    @Published var debugLoggingEnabled = false
    @Published var justCopied: String? = nil
    
    var isRunning: Bool {
        status == .running
    }
    
    var statusText: String {
        "Status: \(status.rawValue)"
    }
    
    var endpointsText: String {
        "Endpoints: \(endpoints.joined(separator: ", "))"
    }
    
    private var timer: Timer?
    private var pollCount = 0
    private var isUpdating = false
    
    private var helperPath: String {
        Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/detector").path
    }
    
    init() {
        // Immediate status check
        self.updateStatus()
        
        // Auto-start on launch with a slight delay to ensure environment is ready
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            self.updateStatus()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                if !self.isRunning && self.status != .starting {
                    self.start()
                }
            }
        }
        
        // Setup menu tracking for smart polling
        setupMenuTracking()
    }
    
    private func setupMenuTracking() {
        NotificationCenter.default.addObserver(
            forName: NSMenu.didBeginTrackingNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            self?.updateStatus()
            self?.updateModelsAndCLI()
            self?.startPolling()
        }
        
        NotificationCenter.default.addObserver(
            forName: NSMenu.didEndTrackingNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            self?.stopPolling()
        }
    }
    
    func startPolling() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateStatus()
        }
    }
    
    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }
    
    private func updateModelsAndCLI() {
        self.updateModels()
        self.updateCLIStatus()
    }
    
    func updateStatus() {
        guard !isUpdating else { return }
        isUpdating = true
        
        DispatchQueue.global(qos: .background).async {
            let statusResult = ShellHelper.run(self.helperPath, arguments: ["status"])
            let output = statusResult.output
            
            // Capture these values to update on main thread
            var newStatus: DetectorStatus = .stopped
            
            if output.contains("Running") {
                newStatus = .running
            } else if output.contains("Starting...") {
                newStatus = .starting
            } else if output.contains("Stopping...") {
                newStatus = .stopping
            } else if output.contains("Restarting...") {
                newStatus = .restarting
            } else {
                newStatus = .stopped
            }
            
            let newStartupEnabled = output.contains("Startup:   enabled")
            let newDebugEnabled = output.contains("Debug:     enabled")
            
            // Parse endpoints
            var newEndpoints: [String] = []
            let lines = output.split(separator: "\n")
            for line in lines {
                if line.contains("Endpoint:"), let addr = line.split(separator: ":", maxSplits: 1).last?.trimmingCharacters(in: .whitespaces) {
                    newEndpoints.append(addr)
                }
            }
            
            // Fallback to defaults if none found (e.g. if stopped)
            if newEndpoints.isEmpty {
                newEndpoints = Constants.defaultEndpoints
            }
            
            DispatchQueue.main.async {
                // Only update status from poll if we are not in a transitional state
                // or if the helper itself reports a transitional state
                self.status = newStatus
                self.startupEnabled = newStartupEnabled
                self.endpoints = newEndpoints
                self.debugLoggingEnabled = newDebugEnabled
                
                self.isUpdating = false
            }
        }
    }
    
    func toggleDetector() {
        if isRunning {
            stop()
        } else {
            start()
        }
    }
    
    func start() {
        if isRunning || status == .starting { return }
        
        status = .starting
        
        DispatchQueue.global(qos: .background).async {
            _ = ShellHelper.run(self.helperPath, arguments: ["start", "--daemon"])
            DispatchQueue.main.async {
                self.updateStatus()
            }
        }
    }
    
    func stop() {
        if status == .stopped || status == .stopping { return }
        
        status = .stopping
        
        DispatchQueue.global(qos: .background).async {
            _ = ShellHelper.run(self.helperPath, arguments: ["stop"])
            DispatchQueue.main.async {
                self.updateStatus()
            }
        }
    }
    
    func restart() {
        status = .restarting
        
        DispatchQueue.global(qos: .background).async {
            _ = ShellHelper.run(self.helperPath, arguments: ["restart"])
            DispatchQueue.main.async {
                self.updateStatus()
            }
        }
    }
    
    func toggleStartup() {
        startupEnabled.toggle()
        let newState = startupEnabled
        
        DispatchQueue.global(qos: .background).async {
            _ = ShellHelper.run(self.helperPath, arguments: ["startup", newState ? "enable" : "disable"])
            DispatchQueue.main.async {
                self.updateStatus()
            }
        }
    }
    
    func toggleDebugLogging() {
        let newState = debugLoggingEnabled
        DispatchQueue.global(qos: .background).async {
            _ = ShellHelper.run(self.helperPath, arguments: ["debug", newState ? "enable" : "disable"])
            DispatchQueue.main.async {
                self.updateStatus()
            }
        }
    }
    
    func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        
        justCopied = text
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            if self.justCopied == text {
                self.justCopied = nil
            }
        }
    }
    
    private func updateModels() {
        var foundModels: [(name: String, sizeMB: Double)] = []
        if let files = try? FileManager.default.contentsOfDirectory(at: Constants.modelsDir, includingPropertiesForKeys: [.fileSizeKey]) {
            for file in files where file.pathExtension == "onnx" {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: file.path),
                   let size = attrs[.size] as? UInt64 {
                    foundModels.append((file.lastPathComponent, Double(size) / (1024 * 1024)))
                }
            }
        }
        models = foundModels.sorted { $0.name < $1.name }
    }
    
    private func updateCLIStatus() {
        cliInstalled = FileManager.default.fileExists(atPath: Constants.cliInstallDir.appendingPathComponent("detector").path) ||
                       FileManager.default.fileExists(atPath: Constants.cliFallbackDir.appendingPathComponent("detector").path)
    }
    
    func viewLogs() {
        let logPath = Constants.logDir.appendingPathComponent("detector.log")
        if FileManager.default.fileExists(atPath: logPath.path) {
            ShellHelper.run("/usr/bin/open", arguments: ["-a", "Console", logPath.path])
        } else {
            showAlert(title: "Log file not found", message: "Ensure the detector has been started.")
        }
    }
    
    func installCLI() {
        let cliBin = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/detector")
        if !FileManager.default.fileExists(atPath: cliBin.path) {
            showAlert(title: "CLI Tool Not Found", message: "The detector binary was not found in the application bundle.")
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let result = ShellHelper.run(self.helperPath, arguments: ["install-cli"])
            DispatchQueue.main.async {
                self.updateCLIStatus()
                let output = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
                self.showAlert(title: "Install CLI Tool", message: output.isEmpty ? "CLI installation failed or produced no output." : output)
            }
        }
    }
    
    func uninstallCLI() {
        DispatchQueue.global(qos: .userInitiated).async {
            let result = ShellHelper.run(self.helperPath, arguments: ["uninstall-cli"])
            DispatchQueue.main.async {
                self.updateCLIStatus()
                let output = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
                self.showAlert(title: "Uninstall CLI Tool", message: output.isEmpty ? "CLI uninstallation failed or produced no output." : output)
            }
        }
    }
    
    func showAlert(title: String, message: String) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = message
            alert.runModal()
        }
    }
    
    func quit() {
        stop()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            NSApplication.shared.terminate(nil)
        }
    }
}
