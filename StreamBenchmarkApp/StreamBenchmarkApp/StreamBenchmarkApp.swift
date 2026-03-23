import SwiftUI
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

// MARK: - Logging

/// Structured logger: writes timestamped entries to Documents/benchmark_log.txt
/// and keeps the latest result in Documents/benchmark_results.txt.
func streamLog(_ msg: String) {
    let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let logFile = docsURL.appendingPathComponent("benchmark_log.txt")
    let ts = ISO8601DateFormatter().string(from: Date())
    let entry = "[\(ts)] \(msg)\n"
    if let handle = try? FileHandle(forWritingTo: logFile) {
        handle.seekToEndOfFile()
        handle.write(entry.data(using: .utf8)!)
        handle.closeFile()
    } else {
        try? entry.write(to: logFile, atomically: true, encoding: .utf8)
    }
}

/// Save a structured error report with full diagnostics
func saveErrorReport(phase: String, error: Error, model: String, extra: [String: String] = [:]) {
    let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let ts = ISO8601DateFormatter().string(from: Date())

    var lines = [String]()
    lines.append("=== MLX Layer-Stream ERROR REPORT ===")
    lines.append("Timestamp: \(ts)")
    lines.append("Model: \(model)")
    lines.append("Device: \(deviceInfoStatic())")
    lines.append("Phase: \(phase)")
    lines.append("Error Type: \(type(of: error))")
    lines.append("Error: \(error)")
    lines.append("Localized: \(error.localizedDescription)")
    lines.append("Memory: active=\(GPU.activeMemory / (1024*1024)) MB, peak=\(GPU.peakMemory / (1024*1024)) MB, cache=\(GPU.cacheMemory / (1024*1024)) MB")
    for (k, v) in extra.sorted(by: { $0.key < $1.key }) {
        lines.append("\(k): \(v)")
    }
    lines.append("=== END ERROR REPORT ===")

    let report = lines.joined(separator: "\n")

    // Write to results file (what gets pulled by the deploy script)
    try? report.write(
        to: docsURL.appendingPathComponent("benchmark_results.txt"),
        atomically: true, encoding: .utf8)

    // Also append to log
    streamLog("ERROR [\(phase)] \(error)")
}

func deviceInfoStatic() -> String {
    #if os(iOS)
    return "\(UIDevice.current.name) (\(UIDevice.current.systemName) \(UIDevice.current.systemVersion))"
    #else
    return ProcessInfo.processInfo.hostName
    #endif
}

// MARK: - Configuration

let kPrompt = "Write a detailed explanation of how neural networks learn through backpropagation."
let kMaxTokens = 200
let kNumRuns = 3
let kCooldownSeconds: UInt64 = 30

struct BenchmarkResult: Identifiable {
    let id = UUID()
    let label: String
    let avgTPS: Double
    let promptTPS: Double
    let allTPS: [Double]
    let tokenCount: Int
    let peakMemoryMB: Double
    let loadTimeSeconds: Double
}

// MARK: - ViewModel

@MainActor
class BenchmarkViewModel: ObservableObject {
    @Published var status: String = "Ready"
    @Published var results: [BenchmarkResult] = []
    @Published var isRunning = false
    @Published var modelLoaded = false

    private var container: ModelContainer?
    private var streamingEngine: StreamingEngine?
    private var modelName: String = "unknown"
    private var loadTimeSeconds: Double = 0
    private var isStreamingMode = false

    func loadModel() async {
        isRunning = true
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let localModelURL = docsURL.appendingPathComponent("model")

        let nameFile = docsURL.appendingPathComponent("model_name.txt")
        if let name = try? String(contentsOf: nameFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           !name.isEmpty {
            modelName = name
        }

        guard FileManager.default.fileExists(atPath: localModelURL.appendingPathComponent("config.json").path) else {
            status = "No model found in Documents/model/"
            saveErrorReport(phase: "pre-load", error: NSError(domain: "StreamBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "config.json not found"]), model: modelName)
            isRunning = false
            return
        }

        let isStreaming = FileManager.default.fileExists(
            atPath: localModelURL.appendingPathComponent("streaming_info.json").path)

        status = "Loading \(modelName)\(isStreaming ? " (streaming)" : "")..."
        streamLog("Loading \(modelName), streaming=\(isStreaming)")
        GPU.clearCache()
        GPU.resetPeakMemory()
        let loadStart = CFAbsoluteTimeGetCurrent()

        if isStreaming {
            do {
                try await loadStreamingModel(modelDir: localModelURL)

                isStreamingMode = true
                loadTimeSeconds = CFAbsoluteTimeGetCurrent() - loadStart
                let loadMemMB = Double(GPU.peakMemory) / (1024 * 1024)
                status = "Streaming: \(modelName) (\(String(format: "%.0f", loadMemMB)) MB)"
                modelLoaded = true
                streamLog("Streaming model loaded: \(modelName), \(String(format: "%.0f", loadMemMB)) MB, \(String(format: "%.1f", loadTimeSeconds))s")
            } catch {
                status = "Streaming load failed: \(error)"
                saveErrorReport(phase: "streaming-load", error: error, model: modelName, extra: [
                    "modelDir": localModelURL.path,
                    "non_layer_exists": "\(FileManager.default.fileExists(atPath: localModelURL.appendingPathComponent("non_layer.safetensors").path))",
                ])
            }
        } else {
            do {
                container = try await LLMModelFactory.shared.loadContainer(
                    configuration: ModelConfiguration(directory: localModelURL)
                )
                loadTimeSeconds = CFAbsoluteTimeGetCurrent() - loadStart
                let loadMemMB = Double(GPU.peakMemory) / (1024 * 1024)
                status = "Loaded \(modelName) (\(String(format: "%.0f", loadMemMB)) MB)"
                modelLoaded = true
                streamLog("Baseline model loaded: \(modelName), \(String(format: "%.0f", loadMemMB)) MB")
            } catch {
                status = "Load failed: \(error)"
                saveErrorReport(phase: "baseline-load", error: error, model: modelName)
            }
        }
        isRunning = false
    }

    /// Load model for streaming: create architecture, quantize, load only non-layer weights.
    ///
    /// Key insight: we can't use loadWeights() because it:
    /// 1. Only quantizes modules whose `.scales` key exists in the weight dict
    ///    (non-layer weights don't have layer `.scales`, so layers stay as Linear)
    /// 2. Calls eval(model) which materializes ALL parameters including random layer weights
    ///
    /// Instead we manually: create model → quantize ALL modules → load non-layer weights → eval only non-layer params.
    private func loadStreamingModel(modelDir: URL) async throws {
        // 1. Read config.json to determine model type and quantization
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        streamLog("Model type: \(baseConfig.modelType)")

        // 2. Create model architecture via type registry
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: baseConfig.modelType)
        streamLog("Model architecture created")

        // 3. Quantize ALL quantizable modules (Linear → QuantizedLinear)
        //    This must happen BEFORE loading weights so shapes match the safetensors format.
        if let plq = baseConfig.perLayerQuantization {
            quantize(model: model as! Module) { path, module in
                plq.quantization(layer: path)?.asTuple
            }
            streamLog("Model quantized (per-layer config)")
        } else if let q = baseConfig.quantization {
            quantize(model: model as! Module,
                     groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            streamLog("Model quantized: \(q.bits)-bit, group=\(q.groupSize)")
        }

        // 4. Load non-layer weights from non_layer.safetensors
        let nonLayerURL = modelDir.appendingPathComponent("non_layer.safetensors")
        var weights = try loadArrays(url: nonLayerURL)
        weights = model.sanitize(weights: weights)
        let params = ModuleParameters.unflattened(weights)
        try (model as! Module).update(parameters: params, verify: .none)

        // Only eval non-layer parameters (embed_tokens, norm, lm_head) — NOT layer weights
        eval(weights.values.map { $0 })
        streamLog("Non-layer weights loaded: \(weights.count) tensors")

        // 5. Read streaming info
        let infoData = try Data(contentsOf: modelDir.appendingPathComponent("streaming_info.json"))
        let info = try JSONSerialization.jsonObject(with: infoData) as! [String: Any]
        let numLayers = info["num_layers"] as! Int
        streamLog("Streaming info: \(numLayers) layers")

        // 6. Create streaming engine directly (no ModelContainer needed)
        let engine = StreamingEngine(
            modelDir: modelDir,
            numLayers: numLayers,
            model: model)
        self.streamingEngine = engine
    }

    func runBenchmark() async {
        if isStreamingMode, let engine = streamingEngine {
            await runStreamingBenchmark(engine)
            return
        }
        guard let container else { return }
        isRunning = true
        results = []

        status = "Cooling down (\(kCooldownSeconds)s)..."
        try? await Task.sleep(nanoseconds: kCooldownSeconds * 1_000_000_000)

        let params = GenerateParameters(maxTokens: kMaxTokens, temperature: 0.0)
        var tpsRuns: [Double] = []
        var lastResult: (genTPS: Double, promptTPS: Double, tokens: Int, peakMB: Double)?

        for run in 1...kNumRuns {
            MLX.eval()
            GPU.clearCache()
            try? await Task.sleep(nanoseconds: 5_000_000_000)

            status = "Run \(run)/\(kNumRuns)..."
            do {
                GPU.clearCache()
                GPU.resetPeakMemory()
                let input = try await container.prepare(input: .init(prompt: kPrompt))
                let stream = try await container.generate(input: input, parameters: params)

                var text = ""
                var info: GenerateCompletionInfo?
                for await g in stream {
                    switch g {
                    case .chunk(let t): text += t
                    case .info(let i): info = i
                    case .toolCall: break
                    }
                }
                let peakMB = Double(GPU.peakMemory) / (1024 * 1024)
                let r = (info!.tokensPerSecond, info!.promptTokensPerSecond, info!.generationTokenCount, peakMB)
                tpsRuns.append(r.0)
                lastResult = r
                status = "Run \(run): \(String(format: "%.1f", r.0)) TPS"
            } catch {
                status = "Error: \(error)"
                saveErrorReport(phase: "baseline-run-\(run)", error: error, model: modelName)
            }
        }

        if let last = lastResult {
            let avg = tpsRuns.reduce(0, +) / Double(tpsRuns.count)
            results.append(BenchmarkResult(
                label: "Baseline", avgTPS: avg, promptTPS: last.1,
                allTPS: tpsRuns, tokenCount: last.2,
                peakMemoryMB: last.3, loadTimeSeconds: loadTimeSeconds
            ))
        }

        status = "Complete!"
        isRunning = false
        saveResults()
    }

    /// Run a single streaming decode pass and return (tps, peakMB, avgLoadMs)
    private func runStreamingPass(_ engine: StreamingEngine, maxTokens: Int, label: String) -> (Double, Double, Double) {
        GPU.clearCache()
        GPU.resetPeakMemory()
        engine.resetStats()

        let inputTokens: [Int32] = [9707]
        var currentInput = MLXArray(inputTokens).reshaped([1, 1])
        let cache = engine.newCache()
        let start = CFAbsoluteTimeGetCurrent()

        for step in 0..<maxTokens {
            status = "\(label): token \(step+1)/\(maxTokens)..."
            let logits = engine.streamingStep(currentInput, cache: cache)
            eval(logits)
            let nextToken = MLX.argMax(logits[0..., (-1)..., 0...], axis: -1)
            eval(nextToken)
            currentInput = nextToken.reshaped([1, 1])
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let tps = Double(maxTokens) / elapsed
        let peakMB = Double(GPU.peakMemory) / (1024 * 1024)
        return (tps, peakMB, engine.avgLoadMs)
    }

    private func runStreamingBenchmark(_ engine: StreamingEngine) async {
        isRunning = true
        results = []
        let maxTokens = 10

        // Read streaming info for hybrid calculation
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelDir = docsURL.appendingPathComponent("model")
        var avgLayerMB = 0
        var nonLayerMB = 0
        if let infoData = try? Data(contentsOf: modelDir.appendingPathComponent("streaming_info.json")),
           let info = try? JSONSerialization.jsonObject(with: infoData) as? [String: Any] {
            avgLayerMB = (info["avg_layer_bytes"] as? Int ?? 0) / (1024 * 1024)
            nonLayerMB = (info["non_layer_size_bytes"] as? Int ?? 0) / (1024 * 1024)
        }

        // Run 1: Full streaming (0% resident)
        status = "Stream (full)..."
        streamLog("=== Full streaming ===")
        engine.residentLayers = []
        let (tps1, peak1, load1) = runStreamingPass(engine, maxTokens: maxTokens, label: "Full stream")
        streamLog(String(format: "Full stream: %.2f TPS, %.0f MB, load=%.0fms/layer", tps1, peak1, load1))
        results.append(BenchmarkResult(
            label: "Full stream", avgTPS: tps1, promptTPS: 0,
            allTPS: [tps1], tokenCount: maxTokens,
            peakMemoryMB: peak1, loadTimeSeconds: loadTimeSeconds))

        // Run 2-N: Hybrid at multiple budget levels
        let budgets = [2000, 3000, 3500, 4000, 5000]
        for budgetMB in budgets {
            let residentCount = StreamingEngine.computeResidentCount(
                budgetMB: budgetMB, nonLayerMB: nonLayerMB,
                avgLayerMB: avgLayerMB, numLayers: engine.numLayers)

            guard residentCount > 0 && residentCount < engine.numLayers else { continue }

            // Skip if same resident count as previous budget
            if let prev = results.last, prev.label.contains("/\(engine.numLayers))") {
                let prevCount = Int(prev.label.components(separatedBy: "(").last?.components(separatedBy: "/").first ?? "0") ?? 0
                if prevCount == residentCount { continue }
            }

            status = "Hybrid \(residentCount)/\(engine.numLayers) (\(budgetMB)MB)..."
            streamLog("=== Hybrid: \(residentCount)/\(engine.numLayers) resident, budget=\(budgetMB)MB ===")
            engine.setupResidentLayers(count: residentCount)
            let (tps, peak, load) = runStreamingPass(engine, maxTokens: maxTokens, label: "Hybrid(\(budgetMB)MB)")
            let streamed = engine.numLayers - residentCount
            streamLog(String(format: "Hybrid(%d/%d, %dMB): %.2f TPS, %.0f MB, load=%.0fms/layer (%d streamed)",
                             residentCount, engine.numLayers, budgetMB, tps, peak, load, streamed))
            results.append(BenchmarkResult(
                label: "Hybrid(\(residentCount)/\(engine.numLayers))", avgTPS: tps, promptTPS: 0,
                allTPS: [tps], tokenCount: maxTokens,
                peakMemoryMB: peak, loadTimeSeconds: loadTimeSeconds))
        }

        // Summary
        let summary = results.map { "\($0.label): \(String(format: "%.2f", $0.avgTPS)) TPS" }.joined(separator: " | ")
        status = summary
        streamLog(summary)

        isRunning = false
        saveResults()
    }

    private func saveResults() {
        let text = resultsSummary
        guard !text.isEmpty else { return }
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        try? text.write(to: docsURL.appendingPathComponent("benchmark_results.txt"), atomically: true, encoding: .utf8)
        streamLog("Results saved")
    }

    var resultsSummary: String {
        guard !results.isEmpty else { return "" }
        var lines = [String]()
        lines.append("=== MLX Layer-Stream Device Benchmark ===")
        lines.append("Model: \(modelName)")
        lines.append("max_tokens=\(kMaxTokens), runs=\(kNumRuns)")
        lines.append("Device: \(deviceInfoStatic())")
        lines.append("")
        for r in results {
            lines.append("Strategy: \(r.label)")
            lines.append("  Avg TPS: \(String(format: "%.1f", r.avgTPS))")
            lines.append("  Prompt TPS: \(String(format: "%.1f", r.promptTPS))")
            lines.append("  Runs: [\(r.allTPS.map { String(format: "%.1f", $0) }.joined(separator: ", "))]")
            lines.append("  Peak Memory: \(String(format: "%.0f", r.peakMemoryMB)) MB")
            lines.append("  Load Time: \(String(format: "%.1f", r.loadTimeSeconds))s")
            lines.append("  Tokens: \(r.tokenCount)")
        }
        lines.append("")
        lines.append("=== END ===")
        return lines.joined(separator: "\n")
    }
}

// MARK: - Views

struct ContentView: View {
    @StateObject private var vm = BenchmarkViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Image(systemName: vm.isRunning ? "hourglass" : (vm.modelLoaded ? "checkmark.circle.fill" : "circle"))
                            .foregroundColor(vm.modelLoaded ? .green : .gray)
                        Text(vm.status).font(.subheadline)
                    }.padding(.horizontal)

                    HStack(spacing: 12) {
                        Button("Load Model") { Task { await vm.loadModel() } }
                            .buttonStyle(.borderedProminent)
                            .disabled(vm.isRunning || vm.modelLoaded)
                        Button("Run Benchmark") { Task { await vm.runBenchmark() } }
                            .buttonStyle(.borderedProminent).tint(.orange)
                            .disabled(vm.isRunning || !vm.modelLoaded)
                    }.padding(.horizontal)

                    if !vm.results.isEmpty {
                        Divider()
                        ForEach(vm.results) { r in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text(r.label).font(.headline)
                                    Spacer()
                                    Text(String(format: "%.1f TPS", r.avgTPS))
                                        .font(.title3).fontWeight(.bold).foregroundColor(.orange)
                                }
                                HStack(spacing: 16) {
                                    Label(String(format: "%.0f MB", r.peakMemoryMB), systemImage: "memorychip")
                                    Label(String(format: "%.1fs load", r.loadTimeSeconds), systemImage: "clock")
                                }.font(.caption).foregroundColor(.secondary)
                            }
                            .padding().background(Color.white).cornerRadius(12)
                            .shadow(color: .black.opacity(0.05), radius: 4)
                            .padding(.horizontal)
                        }

                        Text(vm.resultsSummary)
                            .font(.system(.caption, design: .monospaced))
                            .padding().background(Color.gray.opacity(0.1)).cornerRadius(8)
                            .padding(.horizontal)
                    }
                }.padding(.vertical)
            }
            .navigationTitle("Stream Benchmark")
        }
        .task {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            if !vm.modelLoaded && !vm.isRunning {
                await vm.loadModel()
                if vm.modelLoaded {
                    await vm.runBenchmark()
                }
            }
        }
        #if os(iOS)
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
            Task {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if !vm.modelLoaded && !vm.isRunning {
                    await vm.loadModel()
                    if vm.modelLoaded {
                        await vm.runBenchmark()
                    }
                }
            }
        }
        #endif
    }
}

@main
struct StreamBenchmarkAppEntry: App {
    var body: some Scene { WindowGroup { ContentView() } }
}
