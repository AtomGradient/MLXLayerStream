import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Layer-Streaming Engine for running models larger than device memory.
///
/// Requires pre-split weight files (created by split_weights.py):
///   non_layer.safetensors  — always resident (embed, norm, lm_head)
///   layer_XXXX.safetensors — loaded/freed per inference step
class StreamingEngine {

    let modelDir: URL
    let numLayers: Int
    private let module: Module
    private let qwen35Model: Qwen35TextModel

    var layerLoadTimes: [Double] = []
    var layerUnloadTimes: [Double] = []

    init(modelDir: URL, numLayers: Int, module: Module, qwen35Model: Qwen35TextModel) {
        self.modelDir = modelDir
        self.numLayers = numLayers
        self.module = module
        self.qwen35Model = qwen35Model
    }

    // MARK: - Layer Load/Unload

    func loadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()

        let file = modelDir.appendingPathComponent(String(format: "layer_%04d.safetensors", index))
        guard FileManager.default.fileExists(atPath: file.path) else { return }

        do {
            var weights = try loadArrays(url: file)
            weights = qwen35Model.sanitize(weights: weights)

            let params = ModuleParameters.unflattened(weights)
            try module.update(parameters: params, verify: .none)

            // Eval to materialize from disk
            let layerParams = module.parameters().flattened().filter {
                $0.0.contains("layers.\(index).")
            }
            eval(layerParams.map { $0.1 })
        } catch {
            print("  [StreamingEngine] Failed to load layer \(index): \(error)")
        }

        layerLoadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func unloadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()

        let layerParams = module.parameters().flattened().filter {
            $0.0.contains("layers.\(index).")
        }

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

    /// Run a single streaming forward pass (prefill or decode step).
    func streamingStep(_ inputs: MLXArray, cache: [KVCache]) -> MLXArray {
        return qwen35Model.streamingForward(
            inputs, cache: cache,
            layerLoader: { [weak self] i in self?.loadLayer(i) },
            layerUnloader: { [weak self] i in self?.unloadLayer(i) }
        )
    }

    /// Create KV cache for all layers.
    func newCache() -> [KVCache] {
        qwen35Model.newCache(parameters: nil)
    }

    // MARK: - Stats

    func resetStats() {
        layerLoadTimes = []
        layerUnloadTimes = []
    }

    var avgLoadMs: Double {
        guard !layerLoadTimes.isEmpty else { return 0 }
        return layerLoadTimes.reduce(0, +) / Double(layerLoadTimes.count)
    }

    var avgUnloadMs: Double {
        guard !layerUnloadTimes.isEmpty else { return 0 }
        return layerUnloadTimes.reduce(0, +) / Double(layerUnloadTimes.count)
    }

    var totalIOSeconds: Double {
        (layerLoadTimes.reduce(0, +) + layerUnloadTimes.reduce(0, +)) / 1000
    }
}
