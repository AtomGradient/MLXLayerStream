import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

/// Layer-Streaming Engine for running models larger than device memory.
///
/// Uses placeholder quantization to avoid materializing full-precision layer weights.
/// Loads weights per-layer from pre-split safetensors files.
class StreamingEngine {

    let modelDir: URL
    let numLayers: Int
    let model: Qwen35TextModel

    var layerLoadTimes: [Double] = []
    var layerUnloadTimes: [Double] = []

    init(modelDir: URL, numLayers: Int, model: Qwen35TextModel) {
        self.modelDir = modelDir
        self.numLayers = numLayers
        self.model = model
    }

    // MARK: - Streaming Model Loading

    /// Create a streaming-ready Qwen35TextModel with only non-layer weights loaded.
    ///
    /// Steps:
    /// 1. Read config.json to create model architecture
    /// 2. Placeholder-quantize all Linear modules (tiny arrays, no real computation)
    /// 3. Load non_layer.safetensors weights
    /// 4. Layer weights loaded on-demand during streamingForward()
    static func createStreamingModel(directory: URL) throws -> (Qwen35TextModel, Int) {
        // Read config
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        // Determine quantization params from config
        struct QuantConfig: Codable {
            var quantization: QuantInfo?
            struct QuantInfo: Codable {
                var groupSize: Int
                var bits: Int
            }
        }
        let qConfig = try? decoder.decode(QuantConfig.self, from: configData)
        let groupSize = qConfig?.quantization?.groupSize ?? 64
        let bits = qConfig?.quantization?.bits ?? 4

        // Create model via LLM type registry
        let baseDecoder = JSONDecoder()
        let baseConfig = try baseDecoder.decode(BaseConfiguration.self, from: configData)
        let langModel = try waitForAsync {
            try await LLMModelFactory.shared.typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        }

        guard let qwenModel = langModel as? Qwen35TextModel else {
            throw StreamingError.unsupportedModel("Expected Qwen35TextModel, got \(type(of: langModel))")
        }

        // Placeholder-quantize all Linear modules
        placeholderQuantize(model: qwenModel, groupSize: groupSize, bits: bits)

        // Load non-layer weights
        let nonLayerURL = directory.appendingPathComponent("non_layer.safetensors")
        if FileManager.default.fileExists(atPath: nonLayerURL.path) {
            var weights = try loadArrays(url: nonLayerURL)
            weights = qwenModel.sanitize(weights: weights)
            let params = ModuleParameters.unflattened(weights)
            try (qwenModel as Module).update(parameters: params, verify: .none)
            eval(weights.values.map { $0 })
        }

        // Read layer count from streaming info
        let infoData = try Data(contentsOf: directory.appendingPathComponent("streaming_info.json"))
        let info = try JSONSerialization.jsonObject(with: infoData) as! [String: Any]
        let numLayers = info["num_layers"] as! Int

        return (qwenModel, numLayers)
    }

    // MARK: - Placeholder Quantization

    /// Replace all Linear modules with QuantizedLinear using tiny placeholder arrays.
    /// This avoids the massive memory allocation of MLX.quantized() on random weights.
    private static func placeholderQuantize(model: Module, groupSize: Int, bits: Int) {
        let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
            if m is QuantizedLinear { return nil }
            guard let linear = m as? Linear else { return nil }

            let (outDim, inDim) = linear.shape
            let quantizedCols = inDim * bits / 32
            let numGroups = inDim / groupSize

            let weight = MLXArray.zeros([outDim, quantizedCols], dtype: .uint32)
            let scales = MLXArray.zeros([outDim, numGroups], dtype: .float16)
            let biases = MLXArray.zeros([outDim, numGroups], dtype: .float16)
            let bias = linear.bias != nil ? MLXArray.zeros([outDim]) : nil

            return (path, QuantizedLinear(
                weight: weight, bias: bias, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits
            ))
        }
        model.update(modules: ModuleChildren.unflattened(updates))
    }

    // MARK: - Layer Operations

    func loadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()
        let file = modelDir.appendingPathComponent(String(format: "layer_%04d.safetensors", index))
        guard FileManager.default.fileExists(atPath: file.path) else { return }

        do {
            var weights = try loadArrays(url: file)
            weights = model.sanitize(weights: weights)
            let params = ModuleParameters.unflattened(weights)
            try (model as Module).update(parameters: params, verify: .none)
            eval(weights.values.map { $0 })
        } catch {
            streamLog("  [SE] load layer \(index) failed: \(error)")
        }
        layerLoadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func unloadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()
        let layerParams = (model as Module).parameters().flattened().filter {
            $0.0.contains("layers.\(index).")
        }
        var placeholders = [String: MLXArray]()
        for (key, value) in layerParams {
            placeholders[key] = MLXArray.zeros([1], dtype: value.dtype)
        }
        let params = ModuleParameters.unflattened(placeholders)
        try! (model as Module).update(parameters: params, verify: .none)
        GPU.clearCache()
        layerUnloadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func streamingStep(_ inputs: MLXArray, cache: [KVCache]) -> MLXArray {
        model.streamingForward(
            inputs, cache: cache,
            layerLoader: { [weak self] i in self?.loadLayer(i) },
            layerUnloader: { [weak self] i in self?.unloadLayer(i) }
        )
    }

    func newCache() -> [KVCache] {
        model.newCache(parameters: nil)
    }

    func resetStats() {
        layerLoadTimes = []
        layerUnloadTimes = []
    }

    var avgLoadMs: Double {
        layerLoadTimes.isEmpty ? 0 : layerLoadTimes.reduce(0, +) / Double(layerLoadTimes.count)
    }

    var totalIOSeconds: Double {
        (layerLoadTimes.reduce(0, +) + layerUnloadTimes.reduce(0, +)) / 1000
    }
}

// MARK: - Errors

enum StreamingError: Error, LocalizedError {
    case unsupportedModel(String)
    case loadFailed(String)
    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let t): return "Unsupported: \(t)"
        case .loadFailed(let m): return "Load failed: \(m)"
        }
    }
}

// MARK: - Sync async helper

private func waitForAsync<T>(_ block: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<T, Error>!
    Task.detached {
        do { result = .success(try await block()) }
        catch { result = .failure(error) }
        semaphore.signal()
    }
    semaphore.wait()
    return try result.get()
}
