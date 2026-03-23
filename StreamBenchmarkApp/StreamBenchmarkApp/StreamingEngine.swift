import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Layer-Streaming Engine with flash-moe-inspired optimizations:
/// 1. Background prefetch: pread next layer during GPU eval of current layer
/// 2. Hybrid residency: keep N layers resident, only stream the rest
/// 3. Minimal logging: no per-layer logs in production mode
class StreamingEngine {

    let modelDir: URL
    let numLayers: Int
    let model: any LanguageModel
    let verbose: Bool

    var layerLoadTimes: [Double] = []
    var layerUnloadTimes: [Double] = []
    var layerErrors: [(Int, String)] = []

    // Prefetch state (disabled for now — Apple Silicon unified memory contention)
    var prefetchEnabled: Bool = false

    // Hybrid residency: layers in this set are always loaded, never streamed
    var residentLayers: Set<Int> = []

    init(modelDir: URL, numLayers: Int, model: any LanguageModel, verbose: Bool = false) {
        self.modelDir = modelDir
        self.numLayers = numLayers
        self.model = model
        self.verbose = verbose
    }

    // MARK: - Hybrid Residency Setup

    /// Load the first N layers as resident (always in memory, never unloaded).
    /// Call this BEFORE running streamingStep.
    func setupResidentLayers(count: Int) {
        guard count > 0, count <= numLayers else { return }
        residentLayers = Set(0..<count)
        streamLog("[SE] Loading \(count) resident layers...")

        for i in 0..<count {
            let file = modelDir.appendingPathComponent(String(format: "layer_%04d.safetensors", i))
            guard FileManager.default.fileExists(atPath: file.path) else { continue }
            do {
                var weights = try loadArrays(url: file)
                weights = model.sanitize(weights: weights)
                let params = ModuleParameters.unflattened(weights)
                try (model as! Module).update(parameters: params, verify: .none)
                eval(weights.values.map { $0 })
            } catch {
                streamLog("[SE] resident layer \(i) failed: \(error)")
            }
        }
        streamLog("[SE] \(count) resident layers loaded, mem=\(GPU.activeMemory/(1024*1024))MB")
    }

    /// Compute max resident layers given a memory budget.
    /// budget_mb: total memory budget (e.g. 3500 for 3.5GB)
    /// non_layer_mb: memory used by non-layer weights (embed_tokens, norm, lm_head)
    static func computeResidentCount(budgetMB: Int, nonLayerMB: Int, avgLayerMB: Int, numLayers: Int) -> Int {
        let available = budgetMB - nonLayerMB
        guard available > 0, avgLayerMB > 0 else { return 0 }
        return min(available / avgLayerMB, numLayers)
    }

    // MARK: - Layer Operations

    func loadLayer(_ index: Int) {
        if residentLayers.contains(index) { return }

        let start = CFAbsoluteTimeGetCurrent()
        let file = modelDir.appendingPathComponent(String(format: "layer_%04d.safetensors", index))
        guard FileManager.default.fileExists(atPath: file.path) else {
            layerErrors.append((index, "layer file not found"))
            return
        }

        do {
            var weights = try loadArrays(url: file)
            weights = model.sanitize(weights: weights)
            let params = ModuleParameters.unflattened(weights)
            try (model as! Module).update(parameters: params, verify: .none)
            eval(weights.values.map { $0 })
        } catch {
            let msg = "load layer \(index) failed: \(error)"
            layerErrors.append((index, msg))
            streamLog("  [SE] \(msg)")
        }

        layerLoadTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func unloadLayer(_ index: Int) {
        // Skip if resident (never unloaded)
        if residentLayers.contains(index) { return }

        let start = CFAbsoluteTimeGetCurrent()
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

    var streamedLayerCount: Int {
        numLayers - residentLayers.count
    }
}
