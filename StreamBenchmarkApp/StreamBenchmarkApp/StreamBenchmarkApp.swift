import SwiftUI
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

// MARK: - Logging

func streamLog(_ msg: String) {
    let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let logFile = docsURL.appendingPathComponent("benchmark_results.txt")
    let current = (try? String(contentsOf: logFile, encoding: .utf8)) ?? ""
    try? (current + msg + "\n").write(to: logFile, atomically: true, encoding: .utf8)
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
            isRunning = false
            return
        }

        let isStreaming = FileManager.default.fileExists(
            atPath: localModelURL.appendingPathComponent("streaming_info.json").path)

        status = "Loading \(modelName)\(isStreaming ? " (streaming)" : "")..."
        GPU.clearCache()
        GPU.resetPeakMemory()
        let loadStart = CFAbsoluteTimeGetCurrent()

        if isStreaming {
            do {
                // Create temp dir with non_layer.safetensors → model.safetensors + configs
                let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("s_\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
                let fm = FileManager.default
                for item in try fm.contentsOfDirectory(at: localModelURL, includingPropertiesForKeys: nil) {
                    let name = item.lastPathComponent
                    if name == "non_layer.safetensors" {
                        try fm.createSymbolicLink(at: tempDir.appendingPathComponent("model.safetensors"), withDestinationURL: item)
                    } else if name.hasSuffix(".json") && !name.hasPrefix("layer_") && name != "streaming_info.json" {
                        try fm.createSymbolicLink(at: tempDir.appendingPathComponent(name), withDestinationURL: item)
                    }
                }

                // Load model with only non-layer weights
                let tempContainer = try await LLMModelFactory.shared.loadContainer(
                    configuration: ModelConfiguration(directory: tempDir))
                try? fm.removeItem(at: tempDir)

                // Read streaming info
                let infoData = try Data(contentsOf: localModelURL.appendingPathComponent("streaming_info.json"))
                let info = try JSONSerialization.jsonObject(with: infoData) as! [String: Any]
                let numLayers = info["num_layers"] as! Int

                // Set up streaming engine via container.perform
                let engine: StreamingEngine = try await tempContainer.perform { context in
                    StreamingEngine(
                        modelDir: localModelURL,
                        numLayers: numLayers,
                        model: context.model)
                }
                self.streamingEngine = engine
                self.container = tempContainer

                isStreamingMode = true
                loadTimeSeconds = CFAbsoluteTimeGetCurrent() - loadStart
                let loadMemMB = Double(GPU.peakMemory) / (1024 * 1024)
                status = "Streaming: \(modelName) (\(String(format: "%.0f", loadMemMB)) MB)"
                modelLoaded = true
            } catch {
                status = "Stream load: \(error)"
                saveError("Stream load: \(error)")
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

    private func runStreamingBenchmark(_ engine: StreamingEngine) async {
        isRunning = true
        results = []

        status = "Streaming: starting..."
        GPU.clearCache()
        GPU.resetPeakMemory()
        engine.resetStats()

        // Simple decode test
        let inputTokens: [Int32] = [9707]  // "Hello" token
        var currentInput = MLXArray(inputTokens).reshaped([1, 1])
        let cache = engine.newCache()

        let maxTokens = 10
        let start = CFAbsoluteTimeGetCurrent()

        for step in 0..<maxTokens {
            status = "Streaming: token \(step+1)/\(maxTokens)..."
            let logits = engine.streamingStep(currentInput, cache: cache)
            eval(logits)
            let nextToken = MLX.argMax(logits[0..., (-1)..., 0...], axis: -1)
            eval(nextToken)
            currentInput = nextToken.reshaped([1, 1])
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let tps = Double(maxTokens) / elapsed
        let peakMB = Double(GPU.peakMemory) / (1024 * 1024)

        results.append(BenchmarkResult(
            label: "Streaming", avgTPS: tps, promptTPS: 0,
            allTPS: [tps], tokenCount: maxTokens,
            peakMemoryMB: peakMB, loadTimeSeconds: loadTimeSeconds
        ))

        status = String(format: "Streaming: %.2f TPS, %.0f MB, load=%.0fms/layer", tps, peakMB, engine.avgLoadMs)
        isRunning = false
        saveResults()
    }

    private func saveResults() {
        let text = resultsSummary
        guard !text.isEmpty else { return }
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        try? text.write(to: docsURL.appendingPathComponent("benchmark_results.txt"), atomically: true, encoding: .utf8)
    }

    private func saveError(_ message: String) {
        let text = "=== MLX Layer-Stream Device Benchmark ===\nModel: \(modelName)\nDevice: \(deviceInfo())\nERROR: \(message)\n=== END ==="
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        try? text.write(to: docsURL.appendingPathComponent("benchmark_results.txt"), atomically: true, encoding: .utf8)
    }

    var resultsSummary: String {
        guard !results.isEmpty else { return "" }
        var lines = [String]()
        lines.append("=== MLX Layer-Stream Device Benchmark ===")
        lines.append("Model: \(modelName)")
        lines.append("max_tokens=\(kMaxTokens), runs=\(kNumRuns)")
        lines.append("Device: \(deviceInfo())")
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

    private func deviceInfo() -> String {
        #if os(iOS)
        return "\(UIDevice.current.name) (\(UIDevice.current.systemName) \(UIDevice.current.systemVersion))"
        #else
        return ProcessInfo.processInfo.hostName
        #endif
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
