import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Layer-Streaming Engine — uses only MLXLMCommon types (no MLXLLM import).
///
/// Works with any LanguageModel that implements streamingForward().
/// Loads per-layer weights from pre-split safetensors files.
class StreamingEngine {

    let modelDir: URL
    let numLayers: Int
    let model: any LanguageModel

    var layerLoadTimes: [Double] = []
    var layerUnloadTimes: [Double] = []

    init(modelDir: URL, numLayers: Int, model: any LanguageModel) {
        self.modelDir = modelDir
        self.numLayers = numLayers
        self.model = model
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
            try (model as! Module).update(parameters: params, verify: .none)
            eval(weights.values.map { $0 })
        } catch {
            streamLog("  [SE] load layer \(index): \(error)")
        }
        layerLoadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func unloadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()
        let module = model as! Module
        let layerParams = module.parameters().flattened().filter { $0.0.contains("layers.\(index).") }
        var placeholders = [String: MLXArray]()
        for (key, value) in layerParams {
            placeholders[key] = MLXArray.zeros([1], dtype: value.dtype)
        }
        let params = ModuleParameters.unflattened(placeholders)
        try! module.update(parameters: params, verify: .none)
        GPU.clearCache()
        layerUnloadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    // MARK: - Streaming Forward

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
