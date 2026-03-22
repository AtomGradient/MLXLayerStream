import SwiftUI
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Tokenizers

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
    let extraInfo: String  // streaming stats, etc.
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
    private var streamingTokenizer: (any Tokenizer)?
    private var modelName: String = "unknown"
    private var loadTimeSeconds: Double = 0
    private var isStreamingMode = false

    func loadModel() async {
        isRunning = true
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let localModelURL = docsURL.appendingPathComponent("model")

        // Read model name
        let nameFile = docsURL.appendingPathComponent("model_name.txt")
        if let name = try? String(contentsOf: nameFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           !name.isEmpty {
            modelName = name
        }

        guard FileManager.default.fileExists(atPath: localModelURL.appendingPathComponent("config.json").path) else {
            status = "No model found in Documents/model/"
            isRunning = false
            return
        }

        // Check if this is a streaming model (has streaming_info.json)
        let isStreaming = FileManager.default.fileExists(
            atPath: localModelURL.appendingPathComponent("streaming_info.json").path)

        status = "Loading \(modelName)\(isStreaming ? " (streaming)" : "")..."

        GPU.clearCache()
        GPU.resetPeakMemory()
        let loadStart = CFAbsoluteTimeGetCurrent()

        if isStreaming {
            do {
                try await loadStreamingModel(directory: localModelURL)
                loadTimeSeconds = CFAbsoluteTimeGetCurrent() - loadStart
                let loadMemMB = Double(GPU.peakMemory) / (1024 * 1024)
                isStreamingMode = true
                status = "Streaming: \(modelName) (\(String(format: "%.0f", loadMemMB)) MB resident)"
                modelLoaded = true
            } catch {
                status = "Streaming load failed: \(error)"
                saveError("Streaming load failed: \(error)")
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
            } catch {
                status = "Load failed: \(error)"
                saveError("Load failed: \(error)")
            }
        }
        isRunning = false
    }

    private func loadStreamingModel(directory: URL) async throws {
        // Create temp dir with only non_layer.safetensors (as model.safetensors) + configs
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("stream_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let fm = FileManager.default
        for item in try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            let name = item.lastPathComponent
            if name == "non_layer.safetensors" {
                try fm.createSymbolicLink(
                    at: tempDir.appendingPathComponent("model.safetensors"),
                    withDestinationURL: item)
            } else if name.hasSuffix(".json") && !name.hasPrefix("layer_") && name != "streaming_info.json" {
                try fm.createSymbolicLink(
                    at: tempDir.appendingPathComponent(name),
                    withDestinationURL: item)
            }
        }

        // Load model from temp dir (only non-layer weights)
        let tempContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(directory: tempDir)
        )

        // Read streaming info
        let infoData = try Data(contentsOf: directory.appendingPathComponent("streaming_info.json"))
        let info = try JSONSerialization.jsonObject(with: infoData) as! [String: Any]
        let numLayers = info["num_layers"] as! Int

        // Extract model and tokenizer from container
        // Note: we use the deprecated perform that gives us direct model/tokenizer access
        let (engine, tokenizer): (StreamingEngine, any Tokenizer) = try await tempContainer.perform { context in
            guard let qwenModel = context.model as? Qwen35TextModel else {
                throw StreamingError.unsupportedModel("Not a Qwen35TextModel")
            }

            let engine = StreamingEngine(
                modelDir: directory,
                numLayers: numLayers,
                module: qwenModel,
                qwen35Model: qwenModel
            )
            return (engine, context.tokenizer)
        }

        self.streamingEngine = engine
        self.streamingTokenizer = tokenizer

        try? fm.removeItem(at: tempDir)
    }

    // MARK: - Benchmark Modes

    private func runBaselineSingle() async throws -> (genTPS: Double, promptTPS: Double, tokens: Int, peakMB: Double, output: String) {
        guard let container else { throw StreamingError.loadFailed("No container") }

        GPU.clearCache()
        GPU.resetPeakMemory()

        let input = try await container.prepare(input: .init(prompt: kPrompt))
        let stream = try await container.generate(input: input, parameters: GenerateParameters(maxTokens: kMaxTokens, temperature: 0.0))

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
        return (info!.tokensPerSecond, info!.promptTokensPerSecond, info!.generationTokenCount, peakMB, String(text.prefix(120)))
    }

    private func runStreamingSingle() async throws -> (genTPS: Double, tokens: Int, peakMB: Double, avgLoadMs: Double, avgUnloadMs: Double, ioPct: Double) {
        guard let engine = streamingEngine, let tokenizer = streamingTokenizer else {
            throw StreamingError.loadFailed("No streaming engine")
        }

        GPU.clearCache()
        GPU.resetPeakMemory()
        engine.resetStats()

        let tokens = tokenizer.encode(text: kPrompt)
        var inputIds = MLXArray(tokens).reshaped([1, tokens.count])

        // Prefill
        let cache = engine.newCache()
        var logits = engine.streamingStep(inputIds, cache: cache)
        eval(logits)

        // Decode
        var generated: [Int] = []
        let decodeStart = CFAbsoluteTimeGetCurrent()

        for _ in 0..<kMaxTokens {
            let nextToken = MLX.argMax(logits[0..., (-1)..., 0...], axis: -1)
            eval(nextToken)
            let tokenId = nextToken.item(Int.self)
            generated.append(tokenId)

            // Check for EOS
            if let eosId = tokenizer.eosTokenId, tokenId == eosId { break }

            inputIds = nextToken.reshaped([1, 1])
            logits = engine.streamingStep(inputIds, cache: cache)
            eval(logits)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - decodeStart
        let tps = Double(generated.count) / elapsed
        let peakMB = Double(GPU.peakMemory) / (1024 * 1024)
        let ioPct = engine.totalIOSeconds / elapsed * 100

        return (tps, generated.count, peakMB, engine.avgLoadMs, engine.avgUnloadMs, ioPct)
    }

    func runBenchmark() async {
        isRunning = true
        results = []

        status = "Cooling down (\(kCooldownSeconds)s)..."
        try? await Task.sleep(nanoseconds: kCooldownSeconds * 1_000_000_000)

        if isStreamingMode {
            await runStreamingBenchmark()
        } else {
            await runBaselineBenchmark()
        }

        status = "Complete!"
        isRunning = false
        saveResults()
    }

    private func runBaselineBenchmark() async {
        var tpsRuns: [Double] = []
        var lastResult: (genTPS: Double, promptTPS: Double, tokens: Int, peakMB: Double, output: String)?

        for run in 1...kNumRuns {
            MLX.eval()
            GPU.clearCache()
            try? await Task.sleep(nanoseconds: 5_000_000_000)

            status = "Baseline run \(run)/\(kNumRuns)..."
            do {
                let r = try await runBaselineSingle()
                tpsRuns.append(r.genTPS)
                lastResult = r
                status = "Run \(run): \(String(format: "%.1f", r.genTPS)) TPS"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }

        if let last = lastResult {
            let avg = tpsRuns.reduce(0, +) / Double(tpsRuns.count)
            results.append(BenchmarkResult(
                label: "Baseline",
                avgTPS: avg, promptTPS: last.promptTPS,
                allTPS: tpsRuns, tokenCount: last.tokens,
                peakMemoryMB: last.peakMB,
                loadTimeSeconds: loadTimeSeconds,
                extraInfo: ""
            ))
        }
    }

    private func runStreamingBenchmark() async {
        var tpsRuns: [Double] = []
        var lastLoad = 0.0, lastUnload = 0.0, lastIO = 0.0
        var lastTokens = 0, lastPeak = 0.0

        for run in 1...kNumRuns {
            GPU.clearCache()
            try? await Task.sleep(nanoseconds: 5_000_000_000)

            status = "Streaming run \(run)/\(kNumRuns)..."
            do {
                let r = try await runStreamingSingle()
                tpsRuns.append(r.genTPS)
                lastLoad = r.avgLoadMs
                lastUnload = r.avgUnloadMs
                lastIO = r.ioPct
                lastTokens = r.tokens
                lastPeak = r.peakMB
                status = "Run \(run): \(String(format: "%.1f", r.genTPS)) TPS (I/O: \(String(format: "%.0f", r.ioPct))%)"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }

        if !tpsRuns.isEmpty {
            let avg = tpsRuns.reduce(0, +) / Double(tpsRuns.count)
            results.append(BenchmarkResult(
                label: "Streaming",
                avgTPS: avg, promptTPS: 0,
                allTPS: tpsRuns, tokenCount: lastTokens,
                peakMemoryMB: lastPeak,
                loadTimeSeconds: loadTimeSeconds,
                extraInfo: String(format: "avg_load=%.1fms avg_unload=%.1fms io=%.0f%%",
                                  lastLoad, lastUnload, lastIO)
            ))
        }
    }

    // MARK: - Results Output

    private func saveResults() {
        let text = resultsSummary
        guard !text.isEmpty else { return }
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = docsURL.appendingPathComponent("benchmark_results.txt")
        try? text.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    /// Write error to results file so the automation script can capture it.
    private func saveError(_ message: String) {
        let text = """
        === MLX Layer-Stream Device Benchmark ===
        Model: \(modelName)
        Device: \(deviceInfo())
        ERROR: \(message)
        === END ===
        """
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = docsURL.appendingPathComponent("benchmark_results.txt")
        try? text.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    var resultsSummary: String {
        guard !results.isEmpty else { return "" }
        var lines = [String]()
        lines.append("=== MLX Layer-Stream Device Benchmark ===")
        lines.append("Model: \(modelName)")
        lines.append("max_tokens=\(kMaxTokens), runs=\(kNumRuns)")
        lines.append("Device: \(deviceInfo())")
        lines.append("Mode: \(isStreamingMode ? "STREAMING" : "BASELINE")")
        lines.append("")
        for r in results {
            lines.append("Strategy: \(r.label)")
            lines.append("  Avg TPS: \(String(format: "%.1f", r.avgTPS))")
            lines.append("  Prompt TPS: \(String(format: "%.1f", r.promptTPS))")
            lines.append("  Runs: [\(r.allTPS.map { String(format: "%.1f", $0) }.joined(separator: ", "))]")
            lines.append("  Peak Memory: \(String(format: "%.0f", r.peakMemoryMB)) MB")
            lines.append("  Load Time: \(String(format: "%.1f", r.loadTimeSeconds))s")
            lines.append("  Tokens: \(r.tokenCount)")
            if !r.extraInfo.isEmpty {
                lines.append("  Streaming: \(r.extraInfo)")
            }
        }
        lines.append("")
        lines.append("=== END ===")
        return lines.joined(separator: "\n")
    }

    private func deviceInfo() -> String {
        #if os(iOS)
        return "\(UIDevice.current.name) (\(UIDevice.current.systemName) \(UIDevice.current.systemVersion))"
        #else
        return ProcessInfo.processInfo.hostName
        #endif
    }
}

enum StreamingError: Error, LocalizedError {
    case unsupportedModel(String)
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let t): return "Unsupported model type: \(t)"
        case .loadFailed(let msg): return "Load failed: \(msg)"
        }
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
                                if !r.extraInfo.isEmpty {
                                    Text(r.extraInfo).font(.caption2).foregroundColor(.secondary)
                                }
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
