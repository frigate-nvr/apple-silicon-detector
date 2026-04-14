import SwiftUI
import AppKit

@main
struct FrigateDetectorApp: App {
    @StateObject private var detector = DetectorManager()

    var body: some Scene {
        MenuBarExtra {
            DetectorMenuView(detector: detector)
        } label: {
            Image("frigate")
                .symbolRenderingMode(.hierarchical)
                .font(.system(size: 18, weight: .semibold))
        }
        .menuBarExtraStyle(.menu)
    }
}
