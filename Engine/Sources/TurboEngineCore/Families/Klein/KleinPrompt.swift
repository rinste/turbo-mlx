import Foundation
import MLX
import Tokenizers

/// Turns a prompt into what Klein's text encoder expects: the Qwen3 chat template around it
/// (thinking off), padded to the maximum length, with its attention mask.
public final class KleinPrompter {
    public enum PromptError: LocalizedError {
        case noTokenizer(URL)

        public var errorDescription: String? {
            switch self {
            case .noTokenizer(let url): "No tokenizer found in \(url.path)/tokenizer."
            }
        }
    }

    private let tokenizer: any Tokenizer
    private let chatTemplate: String?
    private let padTokenId: Int
    private let maxLength: Int

    public init(modelPath: URL, maxLength: Int) throws {
        let folder = modelPath.appending(path: "tokenizer", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer.json").path) else {
            throw PromptError.noTokenizer(modelPath)
        }
        tokenizer = try Blocking.run { try await AutoTokenizer.from(modelFolder: folder) }
        // mflux downloads the template next to the tokenizer folder; swift-transformers reads the
        // one inside it (or the config's). Keep the root one as a fallback.
        let rootTemplate = modelPath.appending(path: "chat_template.jinja")
        chatTemplate = (try? String(contentsOf: rootTemplate, encoding: .utf8)).flatMap { $0.isEmpty ? nil : $0 }
        padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? tokenizer.eosTokenId ?? 0
        self.maxLength = maxLength
    }

    /// Token ids padded to `maxLength` and the mask marking the real ones, as `LanguageTokenizer`
    /// builds them with `padding="max_length"`: both [1, maxLength] int32.
    public func tokenize(_ prompt: String) throws -> (inputIds: MLXArray, attentionMask: MLXArray) {
        var ids = try templated(prompt)
        if ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
        let real = ids.count
        ids.append(contentsOf: Array(repeating: padTokenId, count: maxLength - real))
        let mask = (0 ..< maxLength).map { $0 < real ? Int32(1) : Int32(0) }
        return (MLXArray(ids.map { Int32($0) }, [1, maxLength]), MLXArray(mask, [1, maxLength]))
    }

    /// `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
    /// enable_thinking=False)`, tokenized.
    private func templated(_ prompt: String) throws -> [Int] {
        let messages: [[String: any Sendable]] = [["role": "user", "content": prompt]]
        let context: [String: any Sendable] = ["enable_thinking": false]
        if tokenizer.hasChatTemplate {
            return try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: nil, addGenerationPrompt: true, truncation: false,
                maxLength: nil, tools: nil, additionalContext: context
            )
        }
        if let chatTemplate {
            return try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true,
                truncation: false, maxLength: nil, tools: nil, additionalContext: context
            )
        }
        // Qwen3's template for one user turn with thinking disabled.
        let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }
}

/// Runs an async operation to completion from synchronous code (the engine is single-threaded).
enum Blocking {
    static func run<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = ResultBox<T>()
        Task.detached {
            do { box.result = .success(try await operation()) } catch { box.result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        return try box.result!.get()
    }

    private final class ResultBox<T>: @unchecked Sendable {
        var result: Result<T, any Error>?
    }
}
