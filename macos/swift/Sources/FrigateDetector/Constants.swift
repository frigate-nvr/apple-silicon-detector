import Foundation

enum Constants {
    // Shared constants — keep in sync with detector/service_manager.py
    static let serviceLabel = "com.frigate.apple-silicon-detector"
    static let guiServiceLabel = "\(serviceLabel).gui"
    static let bundleIdentifier = "com.frigate.apple-silicon-detector.app"
    
    static let logDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/FrigateDetector")
    static let modelsDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/FrigateDetector/models")
    
    static let launchAgentsDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/LaunchAgents")
    
    static let defaultTCPEndpoint = "tcp://0.0.0.0:5555"
    static let defaultIPCEndpoint = "ipc:///tmp/frigate-detector/zmq_detector"
    static let defaultEndpoints = [defaultTCPEndpoint, defaultIPCEndpoint]
    static let defaultProviders = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    
    static let cliInstallDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".local/bin")
    static let cliFallbackDir = URL(fileURLWithPath: "/usr/local/bin")
}
