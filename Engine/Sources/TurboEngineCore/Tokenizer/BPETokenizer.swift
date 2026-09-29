import Foundation

/// A byte-pair-encoding tokenizer read from Hugging Face's `tokenizer.json`: the two kinds the
/// catalog's text encoders use, byte-level BPE after a regex split (Qwen2 / Qwen3, Ling: GPT-2's
/// byte alphabet) and SentencePiece-style BPE with byte fallback (Gemma 3: spaces as "▁", the
/// text in one piece). Encoding only, as `tokenizers` does it: added tokens are cut out of the raw
/// text first, each stretch between them is normalized, split and turned into symbols, then pairs
/// merge by rank, the lowest first (the leftmost among equals). Tokens are matched by their exact
/// bytes: Swift's `String` would take canonically equivalent tokens for one (see `TokenizerJSON`).
/// `turbo-engine verify-tokenizers` checks it id for id against `transformers` on a corpus
/// (Fixtures/tokenizers.json).
public final class BPETokenizer {
    /// A token as its UTF-8 bytes.
    private typealias Key = [UInt8]

    public enum TokenizerError: LocalizedError {
        case unreadable(URL)
        case unsupported(String)

        public var errorDescription: String? {
            switch self {
            case .unreadable(let url): "Could not read the tokenizer at \(url.path)."
            case .unsupported(let what): "The tokenizer uses \(what), which Turbo MLX does not implement."
            }
        }
    }

    private struct Pair: Hashable {
        let left: Int
        let right: Int
    }

    private struct Merge {
        let rank: Int
        let id: Int
    }

    private enum Normalizer {
        case nfc
        case replace(String, String)
    }

    private enum Split {
        /// `Split` with a regex, its matches and the text between them as pieces ("Isolated").
        case isolated(NSRegularExpression)
        /// `Split` on a string, each delimiter ending the piece before it ("MergedWithPrevious").
        case mergedWithPrevious(String)
    }

    private let vocab: [Key: Int]
    private let merges: [Pair: Merge]
    private let unknownID: Int?
    /// The `<0xHH>` tokens, when unknown characters fall back to their bytes.
    private let byteIDs: [Int]?
    private let normalizers: [Normalizer]
    private let splits: [Split]
    /// Each byte's token in GPT-2's alphabet (bytes as printable characters), when the tokenizer
    /// is byte-level.
    private let byteSymbols: [Int]?
    /// Added tokens by their first scalar, the longest first.
    private let added: [Unicode.Scalar: [(scalars: [Unicode.Scalar], id: Int)]]
    private let addedIDs: [Key: Int]
    /// What the post-processor puts around a sequence with special tokens.
    private let prefix: [Int]
    private let suffix: [Int]
    private var cache: [Key: [Int]] = [:]

    public init(folder: URL) throws {
        let url = folder.appending(path: "tokenizer.json")
        guard let data = try? Data(contentsOf: url), let json = try? TokenizerJSON.parse(data), let model = json["model"]
        else { throw TokenizerError.unreadable(url) }
        guard model["type"]?.string == "BPE" else { throw TokenizerError.unsupported("a \(model["type"]?.string ?? "?") model") }
        guard let rawVocab = model["vocab"]?.members else { throw TokenizerError.unreadable(url) }

        var vocab = [Key: Int](minimumCapacity: rawVocab.count)
        for (token, id) in rawVocab { vocab[Array(token.utf8)] = id.int }

        var addedIDs: [Key: Int] = [:]
        var added: [Unicode.Scalar: [(scalars: [Unicode.Scalar], id: Int)]] = [:]
        for entry in json["added_tokens"]?.array ?? [] {
            guard let content = entry["content"]?.string, let id = entry["id"]?.int, let first = content.unicodeScalars.first
            else { continue }
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] where entry[flag]?.bool == true {
                throw TokenizerError.unsupported("an added token with \(flag)")
            }
            addedIDs[Array(content.utf8)] = id
            added[first, default: []].append((Array(content.unicodeScalars), id))
        }
        for key in added.keys { added[key]!.sort { $0.scalars.count > $1.scalars.count } }

        var merges = [Pair: Merge]()
        let rawMerges = model["merges"]?.array ?? []
        merges.reserveCapacity(rawMerges.count)
        for (rank, merge) in rawMerges.enumerated() {
            let parts: [Key]
            if let pair = merge.array?.compactMap(\.string), pair.count == 2 {
                parts = pair.map { Array($0.utf8) }
            } else if let text = merge.string, let space = text.utf8.firstIndex(of: UInt8(ascii: " ")) {
                parts = [Array(text.utf8[..<space]), Array(text.utf8[text.utf8.index(after: space)...])]
            } else {
                throw TokenizerError.unsupported("a merge written as \(merge)")
            }
            guard let left = vocab[parts[0]], let right = vocab[parts[1]], let merged = vocab[parts[0] + parts[1]] else { continue }
            let pair = Pair(left: left, right: right)
            if merges[pair] == nil { merges[pair] = Merge(rank: rank, id: merged) }
        }

        let unknown = model["unk_token"]?.string.flatMap { vocab[Array($0.utf8)] ?? addedIDs[Array($0.utf8)] }
        var byteIDs: [Int]?
        if model["byte_fallback"]?.bool == true {
            byteIDs = try (0 ..< 256).map { byte in
                guard let id = vocab[Array(String(format: "<0x%02X>", byte).utf8)] else {
                    throw TokenizerError.unsupported("byte fallback without <0x\(byte)>")
                }
                return id
            }
        }

        self.vocab = vocab
        self.merges = merges
        self.unknownID = unknown
        self.byteIDs = byteIDs
        self.added = added
        self.addedIDs = addedIDs
        normalizers = try Self.normalizers(json["normalizer"])
        var byteLevel = false
        splits = try Self.splits(json["pre_tokenizer"], byteLevel: &byteLevel)
        byteSymbols = byteLevel ? Self.gpt2Alphabet().map { vocab[Array($0.utf8)] ?? unknown ?? 0 } : nil
        (prefix, suffix) = try Self.specialTokens(json["post_processor"], ids: { addedIDs[Array($0.utf8)] ?? vocab[Array($0.utf8)] })
    }

    /// The id of a token of the vocabulary or an added token.
    public func id(of token: String) -> Int? {
        addedIDs[Array(token.utf8)] ?? vocab[Array(token.utf8)]
    }

    /// `tokenizer(text, add_special_tokens=...)["input_ids"]`.
    public func encode(_ text: String, addSpecialTokens: Bool) -> [Int] {
        var ids = addSpecialTokens ? prefix : []
        let scalars = Array(text.unicodeScalars)
        var start = 0
        var index = 0
        func flush(_ end: Int) {
            guard end > start else { return }
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars[start ..< end])
            ids += encodeText(String(view))
        }
        while index < scalars.count {
            if let match = added[scalars[index]]?.first(where: { candidate in
                index + candidate.scalars.count <= scalars.count && scalars[index ..< index + candidate.scalars.count].elementsEqual(candidate.scalars)
            }) {
                flush(index)
                ids.append(match.id)
                index += match.scalars.count
                start = index
            } else {
                index += 1
            }
        }
        flush(scalars.count)
        if addSpecialTokens { ids += suffix }
        return ids
    }

    // MARK: Text between added tokens

    private func encodeText(_ text: String) -> [Int] {
        var text = text
        for normalizer in normalizers {
            switch normalizer {
            case .nfc: text = text.precomposedStringWithCanonicalMapping
            case .replace(let pattern, let content): text = text.replacingOccurrences(of: pattern, with: content, options: .literal)
            }
        }
        var pieces = [text]
        for split in splits {
            pieces = pieces.flatMap { Self.split($0, by: split) }
        }
        var ids: [Int] = []
        for piece in pieces where !piece.isEmpty {
            let key = Array(piece.utf8)
            if let cached = cache[key] {
                ids += cached
                continue
            }
            let tokens = merge(symbols(piece))
            cache[key] = tokens
            ids += tokens
        }
        return ids
    }

    /// The piece's first symbols: its bytes in GPT-2's alphabet, or its characters (Unicode
    /// scalars, as Rust's `chars`), each unknown one as its bytes or as the unknown token.
    private func symbols(_ piece: String) -> [Int] {
        if let byteSymbols {
            return piece.utf8.map { byteSymbols[Int($0)] }
        }
        var ids: [Int] = []
        for scalar in piece.unicodeScalars {
            if let id = vocab[Array(String(scalar).utf8)] {
                ids.append(id)
            } else if let byteIDs {
                ids += String(scalar).utf8.map { byteIDs[Int($0)] }
            } else if let unknownID {
                ids.append(unknownID)
            }
        }
        return ids
    }

    /// Merges the pair of lowest rank, the leftmost among equals, until no pair merges.
    private func merge(_ symbols: [Int]) -> [Int] {
        guard symbols.count > 1 else { return symbols }
        var symbols = symbols
        var previous = Array(-1 ..< symbols.count - 1)
        var next = Array(1 ... symbols.count)
        next[symbols.count - 1] = -1
        var alive = [Bool](repeating: true, count: symbols.count)
        var queue = MergeQueue()
        for index in 0 ..< symbols.count - 1 {
            if let merge = merges[Pair(left: symbols[index], right: symbols[index + 1])] { queue.push(rank: merge.rank, position: index) }
        }
        while let (rank, position) = queue.pop() {
            guard alive[position], next[position] >= 0 else { continue }
            let right = next[position]
            guard let merge = merges[Pair(left: symbols[position], right: symbols[right])], merge.rank == rank else { continue }
            symbols[position] = merge.id
            alive[right] = false
            next[position] = next[right]
            if next[right] >= 0 { previous[next[right]] = position }
            if previous[position] >= 0, let left = merges[Pair(left: symbols[previous[position]], right: symbols[position])] {
                queue.push(rank: left.rank, position: previous[position])
            }
            if next[position] >= 0, let after = merges[Pair(left: symbols[position], right: symbols[next[position]])] {
                queue.push(rank: after.rank, position: position)
            }
        }
        return symbols.indices.filter { alive[$0] }.map { symbols[$0] }
    }

    // MARK: Reading tokenizer.json

    private static func normalizers(_ normalizer: TokenizerJSON?) throws -> [Normalizer] {
        guard let normalizer, normalizer.members != nil else { return [] }
        switch normalizer["type"]?.string {
        case "NFC":
            return [.nfc]
        case "Replace":
            guard let pattern = normalizer["pattern"]?["String"]?.string, let content = normalizer["content"]?.string
            else { throw TokenizerError.unsupported("a Replace normalizer with a regex") }
            return [.replace(pattern, content)]
        case "Sequence":
            return try (normalizer["normalizers"]?.array ?? []).flatMap { try normalizers($0) }
        case let other:
            throw TokenizerError.unsupported("the normalizer \(other ?? "?")")
        }
    }

    private static func splits(_ pre: TokenizerJSON?, byteLevel: inout Bool) throws -> [Split] {
        guard let pre, pre.members != nil else { return [] }
        switch pre["type"]?.string {
        case "Sequence":
            return try (pre["pretokenizers"]?.array ?? []).flatMap { try splits($0, byteLevel: &byteLevel) }
        case "ByteLevel":
            guard pre["add_prefix_space"]?.bool != true, pre["use_regex"]?.bool != true
            else { throw TokenizerError.unsupported("a ByteLevel pre-tokenizer with its own regex or a prefix space") }
            byteLevel = true
            return []
        case "Split":
            guard pre["invert"]?.bool != true, let pattern = pre["pattern"] else { throw TokenizerError.unsupported("an inverted Split") }
            let behavior = pre["behavior"]?.string
            if let regex = pattern["Regex"]?.string, behavior == "Isolated" {
                return [.isolated(try NSRegularExpression(pattern: regex))]
            }
            if let string = pattern["String"]?.string, behavior == "MergedWithPrevious" {
                return [.mergedWithPrevious(string)]
            }
            throw TokenizerError.unsupported("a Split with behavior \(behavior ?? "?")")
        case let other:
            throw TokenizerError.unsupported("the pre-tokenizer \(other ?? "?")")
        }
    }

    private static func split(_ text: String, by split: Split) -> [String] {
        switch split {
        case .isolated(let regex):
            let string = text as NSString
            var pieces: [String] = []
            var last = 0
            for match in regex.matches(in: text, range: NSRange(location: 0, length: string.length)) {
                if match.range.location > last { pieces.append(string.substring(with: NSRange(location: last, length: match.range.location - last))) }
                if match.range.length > 0 { pieces.append(string.substring(with: match.range)) }
                last = match.range.location + match.range.length
            }
            if last < string.length { pieces.append(string.substring(from: last)) }
            return pieces
        case .mergedWithPrevious(let delimiter):
            var pieces: [String] = []
            var rest = Substring(text)
            while let range = rest.range(of: delimiter, options: .literal) {
                pieces.append(String(rest[..<range.upperBound]))
                rest = rest[range.upperBound...]
            }
            if !rest.isEmpty { pieces.append(String(rest)) }
            return pieces
        }
    }

    /// The special tokens a `TemplateProcessing` post-processor puts before and after a single
    /// sequence; ByteLevel and none put nothing.
    private static func specialTokens(_ post: TokenizerJSON?, ids: (String) -> Int?) throws -> ([Int], [Int]) {
        guard let post, post.members != nil else { return ([], []) }
        switch post["type"]?.string {
        case "ByteLevel":
            return ([], [])
        case "TemplateProcessing":
            var before: [Int] = []
            var after: [Int] = []
            var seenSequence = false
            for item in post["single"]?.array ?? [] {
                if item["Sequence"] != nil {
                    seenSequence = true
                } else if let token = item["SpecialToken"]?["id"]?.string {
                    guard let id = ids(token) else { throw TokenizerError.unsupported("the special token \(token)") }
                    if seenSequence { after.append(id) } else { before.append(id) }
                }
            }
            return (before, after)
        case let other:
            throw TokenizerError.unsupported("the post-processor \(other ?? "?")")
        }
    }

    /// GPT-2's `bytes_to_unicode`: printable bytes stand for themselves, the others move to 256+.
    private static func gpt2Alphabet() -> [String] {
        var printable = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var codes = printable
        var extra = 0
        for byte in 0 ..< 256 where !printable.contains(byte) {
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
        }
        var alphabet = [String](repeating: "", count: 256)
        for (byte, code) in zip(printable, codes) { alphabet[byte] = String(Unicode.Scalar(UInt32(code))!) }
        return alphabet
    }
}

/// A min-heap of candidate merges by (rank, position).
private struct MergeQueue {
    private var items: [(rank: Int, position: Int)] = []

    private static func less(_ a: (rank: Int, position: Int), _ b: (rank: Int, position: Int)) -> Bool {
        a.rank != b.rank ? a.rank < b.rank : a.position < b.position
    }

    mutating func push(rank: Int, position: Int) {
        items.append((rank, position))
        var child = items.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard Self.less(items[child], items[parent]) else { break }
            items.swapAt(child, parent)
            child = parent
        }
    }

    mutating func pop() -> (Int, Int)? {
        guard let top = items.first else { return nil }
        let last = items.removeLast()
        if !items.isEmpty {
            items[0] = last
            var parent = 0
            while true {
                let left = 2 * parent + 1
                let right = left + 1
                var smallest = parent
                if left < items.count, Self.less(items[left], items[smallest]) { smallest = left }
                if right < items.count, Self.less(items[right], items[smallest]) { smallest = right }
                if smallest == parent { break }
                items.swapAt(parent, smallest)
                parent = smallest
            }
        }
        return (top.rank, top.position)
    }
}
