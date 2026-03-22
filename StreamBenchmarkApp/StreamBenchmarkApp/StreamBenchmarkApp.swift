import SwiftUI
import MLX
import MLXLLM
import MLXLMCommon

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
    let outputPreview: String
}

// MARK: - ViewModel

@MainActor
class BenchmarkViewModel: ObservableObject {
    @Published var status: String = "Ready"
    @Published var results: [BenchmarkResult] = []
    @Published var isRunning = false
    @Published var modelLoaded = false

    private var container: ModelContainer?
    private var modelName: String = "unknown"
    private var loadTimeSeconds: Double = 0

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

        status = "Loading \(modelName)..."

        // Measure load time
        GPU.clearCache()
        GPU.resetPeakMemory()
        let loadStart = CFAbsoluteTimeGetCurrent()

        do {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: ModelConfiguration(directory: localModelURL)
            )
            loadTimeSeconds = CFAbsoluteTimeGetCurrent() - loadStart
            let loadMemMB = Double(GPU.peakMemory) / (1024 * 1024)
            status = "Loaded \(modelName) in \(String(format: "%.1f", loadTimeSeconds))s (\(String(format: "%.0f", loadMemMB)) MB)"
            modelLoaded = true
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
        isRunning = false
    }

    private func runSingle(container: ModelContainer, params: GenerateParameters) async throws -> (genTPS: Double, promptTPS: Double, tokens: Int, peakMB: Double, output: String) {
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
        return (info!.tokensPerSecond, info!.promptTokensPerSecond, info!.generationTokenCount, peakMB, String(text.prefix(120)))
    }

    func runBenchmark() async {
        guard let container else { return }
        isRunning = true
        results = []

        let params = GenerateParameters(maxTokens: kMaxTokens, temperature: 0.0)

        // Cooldown before starting
        status = "Cooling down (\(kCooldownSeconds)s)..."
        try? await Task.sleep(nanoseconds: kCooldownSeconds * 1_000_000_000)

        status = "Running baseline..."
        var tpsRuns: [Double] = []
        var lastResult: (genTPS: Double, promptTPS: Double, tokens: Int, peakMB: Double, output: String)?

        for run in 1...kNumRuns {
            MLX.eval()
            GPU.clearCache()
            try? await Task.sleep(nanoseconds: 5_000_000_000)  // 5s cooldown between runs

            do {
                let r = try await runSingle(container: container, params: params)
                tpsRuns.append(r.genTPS)
                lastResult = r
                status = "Run \(run)/\(kNumRuns): \(String(format: "%.1f", r.genTPS)) TPS"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }

        if let last = lastResult {
            let avg = tpsRuns.reduce(0, +) / Double(tpsRuns.count)
            results.append(BenchmarkResult(
                label: "Baseline",
                avgTPS: avg,
                promptTPS: last.promptTPS,
                allTPS: tpsRuns,
                tokenCount: last.tokens,
                peakMemoryMB: last.peakMB,
                loadTimeSeconds: loadTimeSeconds,
                outputPreview: last.output
            ))
        }

        status = "Complete!"
        isRunning = false
        saveResults()
    }

    private func saveResults() {
        let text = resultsSummary
        guard !text.isEmpty else { return }
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
