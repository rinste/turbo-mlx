import Foundation
import MLX
import Tokenizers

/// Turns a prompt into what a Qwen3 text encoder expects: the chat template around it (thinking
/// on or off, as each family's mflux tokenizer definition asks), tokenized, either padded to the
/// maximum length with its attention mask (FLUX.2 Klein) or just the real tokens (Z-Image).
public final class Qwen3Prompter {
    public enum PromptError: LocalizedError {
        case noTokenizer(URL)

        public var errorDescription: String? {
            switch self {
            case .noTokenizer(let url): "No tokenizer found in \(url.path)."
            }
        }
    }

    private let tokenizer: any Tokenizer
    private let chatTemplate: String?
    private let padTokenId: Int
    private let maxLength: Int
    private let enableThinking: Bool

    /// - Parameters:
    ///   - folder: the tokenizer folder inside the checkpoint (`tokenizer/` for both families).
    ///   - enableThinking: the `enable_thinking` the chat template is rendered with (Klein: off,
    ///     Z-Image: on).
    public init(modelPath: URL, folder: String = "tokenizer", maxLength: Int, enableThinking: Bool = false) throws {
        let tokenizerFolder = modelPath.appending(path: folder, directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: tokenizerFolder.appending(path: "tokenizer.json").path) else {
            throw PromptError.noTokenizer(tokenizerFolder)
        }
        tokenizer = try Blocking.run { try await AutoTokenizer.from(modelFolder: tokenizerFolder) }
        // mflux downloads the template next to the tokenizer folder; swift-transformers reads the
        // one inside it (or the config's). Keep the root one as a fallback.
        let rootTemplate = modelPath.appending(path: "chat_template.jinja")
        chatTemplate = (try? String(contentsOf: rootTemplate, encoding: .utf8)).flatMap { $0.isEmpty ? nil : $0 }
        padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? tokenizer.eosTokenId ?? 0
        self.maxLength = maxLength
        self.enableThinking = enableThinking
    }

    /// Token ids padded to `maxLength` and the mask marking the real ones, as `LanguageTokenizer`
    /// builds them with `padding="max_length"`: both [1, maxLength] int32.
    public func tokenize(_ prompt: String) throws -> (inputIds: MLXArray, attentionMask: MLXArray) {
        let ids = try tokenIds(prompt)
        let real = ids.count
        let padded = ids + Array(repeating: padTokenId, count: maxLength - real)
        let mask = (0 ..< maxLength).map { $0 < real ? Int32(1) : Int32(0) }
        return (MLXArray(padded.map { Int32($0) }, [1, maxLength]), MLXArray(mask, [1, maxLength]))
    }

    /// The real tokens only, truncated to `maxLength` (what the padded form's mask would keep).
    public func tokenIds(_ prompt: String) throws -> [Int] {
        var ids = try templated(prompt)
        if ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
        return ids
    }

    /// `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
    /// enable_thinking=...)`, tokenized.
    private func templated(_ prompt: String) throws -> [Int] {
        let messages: [[String: any Sendable]] = [["role": "user", "content": prompt]]
        let context: [String: any Sendable] = ["enable_thinking": enableThinking]
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
        // Qwen3's template for one user turn: with thinking disabled it closes an empty think block.
        let think = enableThinking ? "" : "<think>\n\n</think>\n\n"
        let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n\(think)"
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }
}

/// A tokenizer read from a folder with `tokenizer.json`, for the families whose prompt is a
/// literal template around the text rather than a chat template (Qwen-Image, Ming-Image).
public final class TemplatePrompter {
    private let tokenizer: any Tokenizer
    private let template: String
    private let maxLength: Int?
    private let addSpecialTokens: Bool

    /// `template` holds `{}` where the prompt goes, as mflux's `TokenizerDefinition.template`.
    public init(folder: URL, template: String, maxLength: Int?, addSpecialTokens: Bool) throws {
        guard FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer.json").path) else {
            throw Qwen3Prompter.PromptError.noTokenizer(folder)
        }
        tokenizer = try Blocking.run { try await AutoTokenizer.from(modelFolder: folder) }
        self.template = template
        self.maxLength = maxLength
        self.addSpecialTokens = addSpecialTokens
    }

    /// The template's tokens with the prompt in place, truncated to the maximum length.
    public func tokenIds(_ prompt: String) -> [Int] {
        let text = template.replacingOccurrences(of: "{}", with: prompt)
        var ids = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        if let maxLength, ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
        return ids
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
