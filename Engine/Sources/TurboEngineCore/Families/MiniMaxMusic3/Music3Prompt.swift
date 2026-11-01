import Foundation

/// The text MiniMax Music 3 reads: the caption cleaned of Markdown, the lyrics with their section tags on lines of their
/// own and lower-cased, both between the checkpoint's special tokens, and its token ids twice: as written, and for the
/// unconditional pass with every token but the first and the last two replaced by the guidance token. This is the
/// official integration's contract (diffusers' `modular_pipelines/minimax_music3/encoders.py`, which mlx-audio's
/// `prompt.py` follows but for a few edge cases); "even whitespace-level changes change the audio", so Python's
/// string rules are followed to the code point: its whitespace, `splitlines`, `strip`, `lower`, and `re` (patterns
/// written with Python's `\s` spelled out).
public enum Music3Prompt {
    public static func assemble(caption: String, lyrics: String) -> String {
        "<|im_start|><|caption_start|>\(cleanCaption(caption))<|caption_end|>"
            + "<|lyrics_start|>\(normalizeLyrics(lyrics))<|lyrics_end|>"
            + "<|im_end|><|audio_start|>"
    }

    /// The ids of the conditional and the unconditional prompt, of equal length.
    public static func idPair(_ ids: [Int], cfgToken: Int) -> (conditional: [Int], unconditional: [Int]) {
        guard ids.count > 3 else { return (ids, ids) }
        return (ids, [ids[0]] + Array(repeating: cfgToken, count: ids.count - 3) + ids.suffix(2))
    }

    // MARK: The caption

    /// `_clean_caption`: `<|key value|>` tags become "key is value", Markdown headings, bullets, bold and italics lose
    /// their marks, lines lose their trailing whitespace, rules ("---") are emptied, bullets "• " and runs of four
    /// spaces go, and blank lines are dropped.
    public static func cleanCaption(_ caption: String) -> String {
        let tagged = PythonText.sub(#"<\|([^|]*)\|>"#, in: Array(caption.unicodeScalars)) { groups in
            let inner = PythonText.strip(groups[1])
            let parts = PythonText.split(inner, maxSplits: 1)
            return parts.count == 2 ? parts[0] + Array(" is ".unicodeScalars) + parts[1] : inner
        }
        let s = PythonText.space
        var lines: [[Unicode.Scalar]] = []
        for var line in PythonText.splitLines(tagged) {
            line = PythonText.sub("^\(s){0,3}#{1,6}\(s)+", in: line, template: "")
            line = PythonText.sub(#"^\#(s)*[*+-]\#(s)+"#, in: line, template: "")
            line = PythonText.sub(#"^\#(s)*\*\#(s)+"#, in: line, template: "")
            // Bold inside bold: again until nothing changes.
            while PythonText.contains(line, "**") {
                let updated = PythonText.sub(#"\*\*([^*]+)\*\*"#, in: line, template: "$1")
                if updated == line { break }
                line = updated
            }
            line = PythonText.sub(#"(?<!\*)\*([^*\n]+)\*(?!\*)"#, in: line, template: "$1")
            lines.append(PythonText.rstrip(line))
        }
        var text = PythonText.join(lines, separator: "\n")
        text = PythonText.sub(#"^\#(s)*[-*_]{3,}\#(s)*$"#, in: text, template: "", multiline: true)
        text = PythonText.replace(text, "• ", "")
        text = PythonText.replace(text, "    ", "")
        text = PythonText.sub(#"\n{2,}"#, in: text, template: "\n")
        return PythonText.string(text)
    }

    // MARK: The lyrics

    /// `_normalize_lyrics`: a line (split at "\n" only) that starts with section tags keeps only them, a tag after a space or followed by one
    /// starts a line of its own, " ^ " breaks a line, tags are lower-cased, and the text starts with "[start]".
    public static func normalizeLyrics(_ lyrics: String) -> String {
        let leadingTags = PythonText.regex(#"^[ \t]*((?:\[[^\]]+\][ \t]*)+)"#)
        var output: [[Unicode.Scalar]] = []
        for line in PythonText.split(Array(lyrics.unicodeScalars), on: "\n") {
            let string = PythonText.string(line) as NSString
            if let match = leadingTags.firstMatch(in: string as String, range: NSRange(location: 0, length: string.length)) {
                output.append(PythonText.strip(Array(string.substring(with: match.range(at: 1)).unicodeScalars)))
            } else {
                output.append(line)
            }
        }
        var text = PythonText.join(output, separator: "\n")
        text = PythonText.replace(text, "] ", "]\n")
        text = PythonText.replace(text, " [", "\n[")
        text = PythonText.replace(text, " ^ ", "\n")
        text = PythonText.sub(#"\[([^\]]+)\]"#, in: text) { groups in
            Array("[".unicodeScalars) + PythonText.lower(groups[1]) + Array("]".unicodeScalars)
        }
        return "[start]\n" + PythonText.string(text)
    }
}

/// Python's `str` and `re` where MiniMax Music 3's prompt needs them, on Unicode scalars.
enum PythonText {
    /// What `str.isspace` accepts (and `re`'s `\s` matches in a `str` pattern), as a regex class.
    static let space = #"[\t\n\x{0B}\f\r\x{1C}-\x{1F} \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]"#

    static func isSpace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09 ... 0x0D, 0x1C ... 0x20, 0x85, 0xA0, 0x1680, 0x2000 ... 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000: true
        default: false
        }
    }

    /// `str.splitlines()`: no empty piece after a final line break.
    static func splitLines(_ text: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var lines: [[Unicode.Scalar]] = []
        var current: [Unicode.Scalar] = []
        var index = 0
        while index < text.count {
            let scalar = text[index]
            switch scalar.value {
            case 0x0A, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029:
                lines.append(current)
                current = []
            case 0x0D:
                lines.append(current)
                current = []
                if index + 1 < text.count, text[index + 1] == "\n" { index += 1 }
            default:
                current.append(scalar)
            }
            index += 1
        }
        if !current.isEmpty { lines.append(current) }
        return lines
    }

    /// `str.split(separator)`: every piece, empty ones included.
    static func split(_ text: [Unicode.Scalar], on separator: Unicode.Scalar) -> [[Unicode.Scalar]] {
        text.split(separator: separator, omittingEmptySubsequences: false).map(Array.init)
    }

    static func contains(_ text: [Unicode.Scalar], _ target: String) -> Bool {
        let target = Array(target.unicodeScalars)
        guard text.count >= target.count else { return false }
        return (0 ... text.count - target.count).contains { Array(text[$0 ..< $0 + target.count]) == target }
    }

    static func strip(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        guard let first = text.firstIndex(where: { !isSpace($0) }) else { return [] }
        let last = text.lastIndex(where: { !isSpace($0) })!
        return Array(text[first ... last])
    }

    static func rstrip(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        guard let last = text.lastIndex(where: { !isSpace($0) }) else { return [] }
        return Array(text[...last])
    }

    /// `str.split(None, maxsplit)`: runs of whitespace separate, none at either end.
    static func split(_ text: [Unicode.Scalar], maxSplits: Int) -> [[Unicode.Scalar]] {
        var parts: [[Unicode.Scalar]] = []
        var index = 0
        while index < text.count {
            while index < text.count, isSpace(text[index]) { index += 1 }
            guard index < text.count else { break }
            if parts.count == maxSplits {
                parts.append(rstrip(Array(text[index...])))
                break
            }
            let start = index
            while index < text.count, !isSpace(text[index]) { index += 1 }
            parts.append(Array(text[start ..< index]))
        }
        return parts
    }

    /// `str.lower()`, code point by code point (its special cases included, as Python applies them).
    static func lower(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        text.flatMap { scalar in Array(scalar.properties.lowercaseMapping.unicodeScalars) }
    }

    static func replace(_ text: [Unicode.Scalar], _ target: String, _ replacement: String) -> [Unicode.Scalar] {
        let target = Array(target.unicodeScalars)
        let replacement = Array(replacement.unicodeScalars)
        var result: [Unicode.Scalar] = []
        var index = 0
        while index < text.count {
            if index + target.count <= text.count, Array(text[index ..< index + target.count]) == target {
                result += replacement
                index += target.count
            } else {
                result.append(text[index])
                index += 1
            }
        }
        return result
    }

    static func join(_ lines: [[Unicode.Scalar]], separator: String) -> [Unicode.Scalar] {
        Array(lines.map(string).joined(separator: separator).unicodeScalars)
    }

    static func string(_ scalars: [Unicode.Scalar]) -> String {
        var string = String.UnicodeScalarView()
        string.append(contentsOf: scalars)
        return String(string)
    }

    static func regex(_ pattern: String, multiline: Bool = false) -> NSRegularExpression {
        // `^` and `$` at line ends only after "\n" (Python's MULTILINE), never another line separator.
        try! NSRegularExpression(pattern: pattern, options: multiline ? [.anchorsMatchLines, .useUnixLineSeparators] : [.useUnixLineSeparators])
    }

    /// `re.sub` with a template ("$1" for the first group).
    static func sub(_ pattern: String, in text: [Unicode.Scalar], template: String, multiline: Bool = false) -> [Unicode.Scalar] {
        let string = string(text)
        let range = NSRange(location: 0, length: (string as NSString).length)
        return Array(regex(pattern, multiline: multiline).stringByReplacingMatches(in: string, range: range, withTemplate: template).unicodeScalars)
    }

    /// `re.sub` with a function of the match's groups (0 the whole match).
    static func sub(_ pattern: String, in text: [Unicode.Scalar], _ transform: ([[Unicode.Scalar]]) -> [Unicode.Scalar]) -> [Unicode.Scalar] {
        let string = string(text) as NSString
        let expression = regex(pattern)
        var result = ""
        var last = 0
        for match in expression.matches(in: string as String, range: NSRange(location: 0, length: string.length)) {
            result += string.substring(with: NSRange(location: last, length: match.range.location - last))
            let groups = (0 ..< match.numberOfRanges).map { index -> [Unicode.Scalar] in
                let range = match.range(at: index)
                return range.location == NSNotFound ? [] : Array(string.substring(with: range).unicodeScalars)
            }
            result += Self.string(transform(groups))
            last = match.range.location + match.range.length
        }
        result += string.substring(from: last)
        return Array(result.unicodeScalars)
    }
}
