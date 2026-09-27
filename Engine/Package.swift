// swift-tools-version: 6.0
import PackageDescription

// The native engine of Turbo MLX: the catalog's families (FLUX.2 Klein, Z-Image Turbo, Qwen-Image,
// Ming-Image) on MLX Swift, speaking the app's JSON protocol. Build it with Xcode or `xcodebuild`
// (the Metal shaders of mlx-swift need them), see README.md.
let package = Package(
    name: "TurboEngine",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "turbo-engine", targets: ["turbo-engine"]),
        .library(name: "TurboEngineCore", targets: ["TurboEngineCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4"),
    ],
    targets: [
        .target(
            name: "TurboEngineCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "turbo-engine",
            dependencies: ["TurboEngineCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
