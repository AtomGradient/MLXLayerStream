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
    var layerErrors: [(Int, String)] = []

    init(modelDir: URL, numLayers: Int, model: any LanguageModel) {
        self.modelDir = modelDir
        self.numLayers = numLayers
        self.model = model
    }

    // MARK: - Layer Operations

    func loadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()
        streamLog("  [SE] loadLayer(\(index)) start, mem=\(GPU.activeMemory/(1024*1024))MB")
        let file = modelDir.appendingPathComponent(String(format: "layer_%04d.safetensors", index))
        guard FileManager.default.fileExists(atPath: file.path) else {
            let msg = "layer_%04d.safetensors not found"
            layerErrors.append((index, msg))
            streamLog("  [SE] \(String(format: msg, index))")
            return
        }

        do {
            streamLog("  [SE] loadLayer(\(index)) reading file...")
            var weights = try loadArrays(url: file)
            streamLog("  [SE] loadLayer(\(index)) read \(weights.count) tensors, sanitizing...")
            weights = model.sanitize(weights: weights)
            streamLog("  [SE] loadLayer(\(index)) sanitized to \(weights.count) tensors, unflattening...")
            let params = ModuleParameters.unflattened(weights)
            streamLog("  [SE] loadLayer(\(index)) updating model params...")
            try (model as! Module).update(parameters: params, verify: .none)
            streamLog("  [SE] loadLayer(\(index)) eval weights...")
            eval(weights.values.map { $0 })
            streamLog("  [SE] loadLayer(\(index)) done, mem=\(GPU.activeMemory/(1024*1024))MB")
        } catch {
            let msg = "load layer \(index) failed: \(error)"
            layerErrors.append((index, msg))
            streamLog("  [SE] \(msg)")
        }
        layerLoadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func unloadLayer(_ index: Int) {
        let start = CFAbsoluteTimeGetCurrent()
        streamLog("  [SE] unloadLayer(\(index)) start")
        let module = model as! Module
        let layerParams = module.parameters().flattened().filter { $0.0.contains("layers.\(index).") }
        var placeholders = [String: MLXArray]()
        for (key, value) in layerParams {
            placeholders[key] = MLXArray.zeros([1], dtype: value.dtype)
        }
        let params = ModuleParameters.unflattened(placeholders)
        do {
            try module.update(parameters: params, verify: .none)
        } catch {
            let msg = "unload layer \(index) failed: \(error)"
            layerErrors.append((index, msg))
            streamLog("  [SE] \(msg)")
        }
        GPU.clearCache()
        streamLog("  [SE] unloadLayer(\(index)) done, mem=\(GPU.activeMemory/(1024*1024))MB")
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
        layerErrors = []
    }

    var avgLoadMs: Double {
        layerLoadTimes.isEmpty ? 0 : layerLoadTimes.reduce(0, +) / Double(layerLoadTimes.count)
    }

    var totalIOSeconds: Double {
        (layerLoadTimes.reduce(0, +) + layerUnloadTimes.reduce(0, +)) / 1000
    }
}
