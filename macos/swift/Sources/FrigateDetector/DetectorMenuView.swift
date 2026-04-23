import SwiftUI

struct DetectorMenuView: View {
    @ObservedObject var detector: DetectorManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            statusHeader
            
            Divider()
            
            controlsSection
            
            Divider()
            
            modelsSection
            
            endpointsSection
            
            Divider()
            
            advancedSection
            
            Divider()
            
            commonSection
        }
        .frame(minWidth: 220)
    }
    
    // MARK: - Subviews
    
    private var statusHeader: some View {
        Text(detector.statusText)
            .font(.headline)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .disabled(true)
            .accessibilityLabel(detector.statusText)
    }
    
    private var controlsSection: some View {
        Group {
            Button(action: {
                detector.toggleDetector()
            }) {
                Text(detector.isRunning ? "Stop Detector" : "Start Detector")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .disabled(isTransitional)
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .accessibilityLabel(detector.isRunning ? "Stop detector service" : "Start detector service")
            
            Button(action: {
                detector.restart()
            }) {
                Text(detector.status == .restarting ? "Restarting..." : "Restart Detector")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .disabled(!detector.isRunning || isTransitional)
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .accessibilityLabel("Restart detector service")
        }
    }
    
    private var modelsSection: some View {
        Menu("Models") {
            if detector.models.isEmpty {
                Text("No models installed")
                    .disabled(true)
            } else {
                ForEach(detector.models, id: \.name) { model in
                    Text("\(model.name) (\(String(format: "%.1f", model.sizeMB)) MB)")
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 3)
    }
    
    private var endpointsSection: some View {
        Menu("Endpoints") {
            if detector.endpoints.isEmpty {
                Text("No active endpoints")
                    .disabled(true)
            } else {
                ForEach(detector.endpoints, id: \.self) { endpoint in
                    Button(action: {
                        detector.copyToClipboard(endpoint)
                    }) {
                        HStack {
                            Text(endpoint)
                            if #available(macOS 11.0, *) {
                                Spacer()
                                Image(systemName: detector.justCopied == endpoint ? "checkmark.circle.fill" : "doc.on.doc")
                            }
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 3)
    }
    
    private var advancedSection: some View {
        Menu("Advanced") {
            Toggle(isOn: Binding(
                get: { detector.startupEnabled },
                set: { _ in detector.toggleStartup() }
            )) {
                Text("Open at Login")
            }
            .disabled(isTransitional)
            
            Button("View Logs") {
                detector.viewLogs()
            }
            
            Toggle(isOn: Binding(
                get: { detector.debugLoggingEnabled },
                set: { detector.debugLoggingEnabled = $0; detector.toggleDebugLogging() }
            )) {
                Text("Enable Debug Logging")
            }
            
            Divider()
            
            Button(detector.cliInstalled ? "Uninstall CLI Tool" : "Install CLI Tool...") {
                if detector.cliInstalled {
                    detector.uninstallCLI()
                } else {
                    detector.installCLI()
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 3)
    }
    
    private var commonSection: some View {
        Group {
            Button("About") {
                let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
                detector.showAlert(title: "About", message: "Version \(version)\nApple Silicon ONNX Inference\nhttps://github.com/frigate-nvr/apple-silicon-detector")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            
            Button("Quit") {
                detector.quit()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
        }
    }
    
    // MARK: - Helpers
    
    private var isTransitional: Bool {
        detector.status == .starting || detector.status == .stopping || detector.status == .restarting
    }
}
