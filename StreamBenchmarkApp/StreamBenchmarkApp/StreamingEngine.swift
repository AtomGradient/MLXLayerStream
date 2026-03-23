import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Platform Memory Query

/// Get available memory in bytes. iOS uses os_proc_available_memory(), macOS uses host_statistics64.
func getAvailableMemoryBytes() -> UInt64 {
    #if os(iOS) || os(tvOS) || os(watchOS)
    return UInt64(os_proc_available_memory())
    #else
    // macOS: use host_statistics64 to get free + inactive memory
    var stats = vm_statistics64_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    guard result == KERN_SUCCESS else {
        return UInt64(ProcessInfo.processInfo.physicalMemory / 2)  // fallback: half physical
    }
    let pageSize = UInt64(vm_kernel_page_size)
    return (UInt64(stats.free_count) + UInt64(stats.inactive_count)) * pageSize
    #endif
}

// MARK: - Adaptive Residency Configuration

/// Priority mode for adaptive residency: trades off TPS vs context length.
struct AdaptiveResidencyConfig {
    enum Priority: String, CaseIterable {
        case maxTPS     // Minimize KV reserve → maximize resident layers → best TPS
        case balanced   // 1024-token KV reserve → good TPS + reasonable context
        case maxContext // Full KV reserve → fewest resident layers → longest context
    }

    let priority: Priority
    let expectedMaxTokens: Int  // User-specified max context, default 2048

    /// Default KV reserve tokens for each priority mode.
    var kvReserveTokens: Int {
        switch priority {
        case .maxTPS:     return 512
        case .balanced:   return 1024
        case .maxContext: return expectedMaxTokens
        }
    }

    init(priority: Priority = .balanced, expectedMaxTokens: Int = 2048) {
        self.priority = priority
        self.expectedMaxTokens = expectedMaxTokens
    }
}

/// KV cache configuration extracted from model config.json.
struct KVCacheConfig {
    let numKeyValueHeads: Int
    let headDim: Int
    let numLayers: Int
    let fullAttentionInterval: Int  // Qwen3.5 hybrid: only 1/N layers have full KV

    /// Bytes per token for KV cache across all layers.
    /// Each attention layer stores K and V: 2 × numKVHeads × headDim × dtype_size
    /// For hybrid models, only 1/fullAttentionInterval layers have full attention KV.
    var kvBytesPerToken: Int {
        let attentionLayers: Int
        if fullAttentionInterval > 0 {
            attentionLayers = numLayers / fullAttentionInterval
        } else {
            attentionLayers = numLayers
        }
        // 2 bytes per element (bf16/fp16), K + V = 2 tensors
        let bytesPerLayerPerToken = 2 * numKeyValueHeads * headDim * 2
        return bytesPerLayerPerToken * attentionLayers
    }

    static let unknown = KVCacheConfig(numKeyValueHeads: 0, headDim: 0, numLayers: 0, fullAttentionInterval: 0)
}

/// Layer-Streaming Engine with flash-moe-inspired optimizations:
/// 1. Background prefetch: pread next layer during GPU eval of current layer
/// 2. Hybrid residency: keep N layers resident, only stream the rest
/// 3. Adaptive residency: auto-compute optimal resident count based on device memory
/// 4. Dynamic shedding: release layers under memory pressure
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

    // Adaptive residency state
    var kvConfig: KVCacheConfig = .unknown
    var avgLayerMB: Int = 0
    var nonLayerMB: Int = 0
    var tokensSinceLastCheck: Int = 0

    init(modelDir: URL, numLayers: Int, model: any LanguageModel, verbose: Bool = false) {
        self.modelDir = modelDir
        self.numLayers = numLayers
        self.model = model
        self.verbose = verbose
    }

    // MARK: - Adaptive Residency

    /// Compute optimal resident layer count based on available memory, KV needs, and priority.
    static func computeOptimalResidency(
        config: AdaptiveResidencyConfig,
        numLayers: Int,
        avgLayerMB: Int,
        nonLayerMB: Int,
        kvConfig: KVCacheConfig
    ) -> (residentCount: Int, kvReserveMB: Int, layerBudgetMB: Int, availableMB: Int) {
        // Get available memory from OS
        let availableMB = Int(getAvailableMemoryBytes() / (1024 * 1024))

        // Fixed overhead: non-layer weights + 1 streaming layer buffer + runtime overhead
        let runtimeOverheadMB = 150
        let fixedMB = nonLayerMB + avgLayerMB + runtimeOverheadMB

        // KV cache reserve
        let kvReserveMB: Int
        if kvConfig.kvBytesPerToken > 0 {
            kvReserveMB = config.kvReserveTokens * kvConfig.kvBytesPerToken / (1024 * 1024)
        } else {
            // Fallback: estimate 0.5 MB per 100 tokens (conservative)
            kvReserveMB = config.kvReserveTokens * 5 / 1000
        }

        // Layer budget = available - fixed - KV reserve
        let layerBudgetMB = availableMB - fixedMB - kvReserveMB

        guard layerBudgetMB > 0, avgLayerMB > 0 else {
            return (0, kvReserveMB, layerBudgetMB, availableMB)
        }

        // 0.9 safety margin to avoid memory pressure
        let rawCount = layerBudgetMB / avgLayerMB
        let safeCount = Int(Double(rawCount) * 0.9)
        let residentCount = min(max(safeCount, 0), numLayers)

        return (residentCount, kvReserveMB, layerBudgetMB, availableMB)
    }

    /// Predict TPS based on hardware parameters and residency ratio.
    /// Formula: TPS = 1 / (T_compute + N_streamed × T_io_per_layer)
    static func predictTPS(
        numLayers: Int,
        residentCount: Int,
        avgLayerMB: Int,
        totalModelMB: Int,
        memBandwidthGBps: Double,  // GPU memory bandwidth
        nvmeBandwidthGBps: Double  // NVMe read bandwidth
    ) -> Double {
        let streamedLayers = numLayers - residentCount
        // T_compute: time to run the full model from GPU memory
        let tCompute = Double(totalModelMB) / (memBandwidthGBps * 1024)
        // T_io_per_layer: time to load one layer from NVMe
        let tIOPerLayer = Double(avgLayerMB) / (nvmeBandwidthGBps * 1024)

        let totalTime = tCompute + Double(streamedLayers) * tIOPerLayer
        return totalTime > 0 ? 1.0 / totalTime : 0
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

    // MARK: - Dynamic Layer Shedding

    /// Shed resident layers under memory pressure.
    /// Removes from highest index first (front layers more important for representation quality).
    func shedResidentLayers(fraction: Double = 0.25) {
        guard !residentLayers.isEmpty else { return }
        let toShed = max(1, Int(Double(residentLayers.count) * fraction))
        let sortedResident = residentLayers.sorted().reversed()

        var shedCount = 0
        for layerIdx in sortedResident {
            guard shedCount < toShed else { break }
            unloadLayer(layerIdx, force: true)
            residentLayers.remove(layerIdx)
            shedCount += 1
        }
        streamLog("[SE] Shed \(shedCount) resident layers, remaining=\(residentLayers.count), mem=\(GPU.activeMemory/(1024*1024))MB")
    }

    /// Check memory pressure and shed layers if needed.
    /// Call periodically during generation (e.g. every 64 tokens).
    func checkMemoryPressure() {
        tokensSinceLastCheck += 1
        guard tokensSinceLastCheck >= 64 else { return }
        tokensSinceLastCheck = 0

        let availableMB = Int(getAvailableMemoryBytes() / (1024 * 1024))
        if availableMB < 300 {
            streamLog("[SE] Memory pressure! available=\(availableMB)MB, shedding 25% resident layers")
            shedResidentLayers(fraction: 0.25)
        }
    }

    /// Emergency: shed ALL resident layers (called on iOS memory warning).
    func shedAllResidentLayers() {
        guard !residentLayers.isEmpty else { return }
        streamLog("[SE] EMERGENCY: shedding all \(residentLayers.count) resident layers")
        let all = residentLayers.sorted().reversed()
        for layerIdx in all {
            unloadLayer(layerIdx, force: true)
        }
        residentLayers.removeAll()
        GPU.clearCache()
        streamLog("[SE] All layers shed, mem=\(GPU.activeMemory/(1024*1024))MB")
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

    func unloadLayer(_ index: Int, force: Bool = false) {
        // Skip if resident (unless force=true for shedding)
        if !force && residentLayers.contains(index) { return }

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
        checkMemoryPressure()
        return model.streamingForward(
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
