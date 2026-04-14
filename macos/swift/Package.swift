// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FrigateDetector",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "FrigateDetector",
            path: "Sources/FrigateDetector",
            exclude: ["Resources/Info.plist"],
            resources: [.process("Resources")],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/FrigateDetector/Resources/Info.plist"
                ])
            ]
        )
    ]
)
