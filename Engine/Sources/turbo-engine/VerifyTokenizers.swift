import Foundation
import TurboEngineCore

/// `turbo-engine verify-tokenizers`: every family's prompt pipeline, run by the engine's own
/// tokenizer on the corpus of Fixtures/tokenizers.json, against the ids `transformers` gave for it
/// (Fixtures/make_tokenizer_corpus.py). The tokenizers are the catalog's, read from the Hugging Face
/// cache at the commits the corpus names; a pipeline whose checkpoint is not there is skipped.
enum VerifyTokenizers {
    static func run(corpus url: URL) -> Bool {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let prompts = json["prompts"] as? [String],
              let pipelines = json["pipelines"] as? [[String: Any]]
        else {
            print("cannot read \(url.path)")
            return false
        }
        let hub = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
        var ok = true
        for pipeline in pipelines {
            guard let name = pipeline["name"] as? String, let repo = pipeline["repo"] as? String,
                  let commit = pipeline["commit"] as? String, let expected = pipeline["ids"] as? [[Int]]
            else { continue }
            let snapshot = hub.appending(path: "models--" + repo.replacingOccurrences(of: "/", with: "--"))
                .appending(path: "snapshots").appending(path: commit)
            let folder = (pipeline["folder"] as? String).map { $0.isEmpty ? snapshot : snapshot.appending(path: $0) } ?? snapshot
            guard FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer.json").path) else {
                print("· \(name): skipped, \(repo) is not downloaded")
                continue
            }
            let started = Date()
            let tokenize: (String) throws -> [Int]
            do {
                tokenize = try pipelineFunction(name, snapshot: snapshot, folder: folder)
            } catch {
                print("✗ \(name): \(error.localizedDescription)")
                ok = false
                continue
            }
            let loaded = Date().timeIntervalSince(started)
            var failures = 0
            var total = 0
            for (prompt, want) in zip(prompts, expected) {
                let got = (try? tokenize(prompt)) ?? []
                total += want.count
                guard got != want else { continue }
                failures += 1
                if failures <= 3 {
                    let first = (0 ..< min(got.count, want.count)).first { got[$0] != want[$0] } ?? min(got.count, want.count)
                    let shown = prompt.count > 60 ? String(prompt.prefix(60)) + "…" : prompt
                    print("✗ \(name) on \(shown.debugDescription): \(got.count) ids against \(want.count), first difference at \(first):"
                          + " got \(Array(got[min(first, got.count) ..< min(first + 6, got.count)]))"
                          + " want \(Array(want[min(first, want.count) ..< min(first + 6, want.count)]))")
                }
            }
            let seconds = Date().timeIntervalSince(started)
            if failures == 0 {
                print(String(format: "✓ %@: %d prompts, %d ids, identical (loaded in %.2f s, all in %.2f s)", name, prompts.count, total, loaded, seconds))
            } else {
                print("✗ \(name): \(failures) of \(prompts.count) prompts differ")
                ok = false
            }
        }
        print(ok ? "OK: the tokenizers match transformers on the corpus" : "FAILED: see above")
        return ok
    }

    /// The pipeline as the families build it, without truncation.
    private static func pipelineFunction(_ name: String, snapshot: URL, folder: URL) throws -> (String) throws -> [Int] {
        switch name {
        case "qwen3-chat-no-thinking", "qwen3-chat-thinking":
            let prompter = try Qwen3Prompter(modelPath: snapshot, folder: folder.lastPathComponent, maxLength: .max,
                                              enableThinking: name == "qwen3-chat-thinking")
            return { try prompter.tokenIds($0) }
        case "qwen-image":
            let prompter = try TemplatePrompter(folder: folder, template: QwenImageConfig.template, maxLength: nil, addSpecialTokens: true)
            return { prompter.tokenIds($0) }
        case "qwen-image-edit":
            let prompter = try TemplatePrompter(folder: folder, template: "{}", maxLength: nil, addSpecialTokens: true)
            return { prompter.tokenIds(QwenImageConfig.editText(prompt: $0, imageTokens: 16)) }
        case "ming":
            let prompter = try TemplatePrompter(folder: folder, template: MingConfig.promptTemplate, maxLength: nil, addSpecialTokens: false)
            return { prompter.tokenIds($0) }
        case "gemma3":
            let prompter = try Gemma3Prompter(folder: folder, maxLength: .max)
            return { prompter.tokenIds($0) }
        case "sensenova":
            let prompter = try SenseNovaPrompter(folder: folder)
            return { prompter.tokenIds($0) }
        case "sensenova-unconditional":
            let prompter = try SenseNovaPrompter(folder: folder)
            return { _ in prompter.unconditionalIds }
        default:
            throw BPETokenizer.TokenizerError.unsupported("the pipeline \(name)")
        }
    }
}
