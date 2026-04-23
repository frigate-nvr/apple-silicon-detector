import Foundation

enum ShellHelper {
    private static var logURL: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logDir = home.appendingPathComponent("Library/Logs/FrigateDetector")
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        return logDir.appendingPathComponent("detector.log")
    }

    private static let logFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter
    }()

    private static func log(_ message: String) {
        // Always log ERR: and CRASH: but only log others if debug mode is enabled
        let isDebug = UserDefaults.standard.bool(forKey: "EnableDebugLogging")
        let isError = message.contains("ERR:") || message.contains("CRASH:")
        
        if !isDebug && !isError {
            return
        }

        let timestamp = logFormatter.string(from: Date())
        let logMessage = "[\(timestamp)] [gui] \(message)\n"
        
        if let data = logMessage.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logURL.path) {
                if let fileHandle = try? FileHandle(forWritingTo: logURL) {
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(data)
                    fileHandle.closeFile()
                }
            } else {
                try? data.write(to: logURL)
            }
        }
    }

    @discardableResult
    static func run(_ executable: String, arguments: [String] = []) -> (exitCode: Int32, output: String, error: String) {
        log("EXEC: \(executable) \(arguments.joined(separator: " "))")
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            
            let output = String(data: outputData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let error = String(data: errorData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            
            log("EXIT: \(process.terminationStatus)")
            if !output.isEmpty { log("OUT: \(output)") }
            if !error.isEmpty { log("ERR: \(error)") }
            
            return (process.terminationStatus, output, error)
        } catch {
            log("CRASH: \(error.localizedDescription)")
            return (-1, "", error.localizedDescription)
        }
    }
}
