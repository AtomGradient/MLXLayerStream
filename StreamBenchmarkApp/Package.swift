// swift-tools-version: 5.12

import PackageDescription

let package = Package(
    name: "StreamBenchmarkApp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    dependencies: [
        .package(path: "../../H2OAttnScore/mlx-swift"),
        .package(path: "../../H2OAttnScore/mlx-swift-lm"),
    ],
    targets: [
        .executableTarget(
            name: "StreamBenchmarkApp",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "StreamBenchmarkApp"
        ),
    ]
)
