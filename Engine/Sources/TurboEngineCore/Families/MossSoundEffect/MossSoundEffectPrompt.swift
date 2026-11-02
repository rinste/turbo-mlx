import Foundation

/// The text MOSS-SoundEffect's text encoder reads, as OpenMOSS's pipeline makes it: `MossSoundEffectPipeline` strips the
/// prompt and appends " duration: {seconds:.1f}s" (the training captions' convention), then `WanPrompter`'s tokenizer
/// cleans it (`clean='whitespace'`): `ftfy.fix_text`, `html.unescape` twice, Python's strip, runs of `\s` (the `regex`
/// module's) made one space. The port follows ftfy 6.3 step by step except its mojibake repair (`fix_encoding`, which
/// leaves well-formed text alone), and decodes HTML references by number and the named ones of the HTML/XML basics
/// (amp, lt, gt, quot, apos, nbsp, with ftfy's upper-case forms), not the rest of HTML5's 2,231 names.
public enum MossSoundEffectPrompt {
    /// The pipeline's prompt for a clip of `seconds` (rounded to tenths, as `round(seconds, 1)` and `:.1f` do).
    public static func withDuration(_ prompt: String, seconds: Double) -> String {
        PythonText.string(PythonText.strip(Array(prompt.unicodeScalars))) + " duration: " + String(format: "%.1f", seconds) + "s"
    }

    /// `whitespace_clean(basic_clean(text))`.
    public static func clean(_ text: String) -> String {
        let scalars = PythonText.strip(htmlUnescape(htmlUnescape(fixText(Array(text.unicodeScalars)))))
        return PythonText.string(PythonText.strip(collapseSpaces(scalars)))
    }

    /// `re.sub(r'\s+', ' ', text)` with the `regex` module, whose `\s` is Unicode's White_Space (not U+1C–U+1F).
    static func collapseSpaces(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var inRun = false
        for scalar in text {
            if isRegexSpace(scalar) {
                if !inRun { result.append(" ") }
                inRun = true
            } else {
                result.append(scalar)
                inRun = false
            }
        }
        return result
    }

    static func isRegexSpace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09 ... 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000 ... 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000: true
        default: false
        }
    }

    // MARK: ftfy.fix_text

    /// `ftfy.fix_text` with its default configuration, without `fix_encoding`: the text in segments that end after
    /// each "\n", each fixed until nothing changes; HTML is not unescaped from the first segment with a "<" on.
    static func fixText(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var unescape = true
        var start = 0
        while start < text.count {
            var end = start
            while end < text.count, text[end] != "\n" { end += 1 }
            end = min(end + 1, text.count)
            var segment = Array(text[start ..< end])
            if segment.contains("<") { unescape = false }
            while true {
                let original = segment
                if unescape { segment = ftfyUnescape(segment) }
                segment = fixC1Controls(segment)
                segment = segment.flatMap { ligatures[$0.value].map { Array($0.unicodeScalars) } ?? [$0] }
                segment = segment.flatMap { width($0) }
                segment = segment.map { quote($0) }
                segment = fixLineBreaks(segment)
                segment = removeTerminalEscapes(segment)
                segment = segment.filter { !isControl($0) }
                segment = Array(PythonText.string(segment).precomposedStringWithCanonicalMapping.unicodeScalars)
                if segment == original { break }
            }
            result += segment
            start = end
        }
        return result
    }

    /// `fix_c1_controls`: U+80–U+9F read as the Windows-1252 bytes they are (the five it leaves undefined stay).
    static func fixC1Controls(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        text.map { scalar in
            guard (0x80 ... 0x9F).contains(scalar.value) else { return scalar }
            return windows1252[Int(scalar.value - 0x80)]
        }
    }

    static let windows1252: [Unicode.Scalar] = [
        "\u{20AC}", "\u{81}", "\u{201A}", "\u{192}", "\u{201E}", "\u{2026}", "\u{2020}", "\u{2021}",
        "\u{2C6}", "\u{2030}", "\u{160}", "\u{2039}", "\u{152}", "\u{8D}", "\u{17D}", "\u{8F}",
        "\u{90}", "\u{2018}", "\u{2019}", "\u{201C}", "\u{201D}", "\u{2022}", "\u{2013}", "\u{2014}",
        "\u{2DC}", "\u{2122}", "\u{161}", "\u{203A}", "\u{153}", "\u{9D}", "\u{17E}", "\u{178}",
    ]

    /// `LIGATURES`: Latin ligatures and legacy digraphs taken apart.
    static let ligatures: [UInt32: String] = [
        0x132: "IJ", 0x133: "ij", 0x149: "\u{2BC}n", 0x1F1: "DZ", 0x1F2: "Dz", 0x1F3: "dz", 0x1C4: "D\u{17D}", 0x1C5: "D\u{17E}",
        0x1C6: "d\u{17E}", 0x1C7: "LJ", 0x1C8: "Lj", 0x1C9: "lj", 0x1CA: "NJ", 0x1CB: "Nj", 0x1CC: "nj", 0xFB00: "ff",
        0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "\u{17F}t", 0xFB06: "st",
    ]

    /// `WIDTH_MAP`: the ideographic space to a space, and U+FF01–U+FFEF to their NFKC forms (fullwidth ASCII and
    /// punctuation, halfwidth katakana and Hangul).
    static func width(_ scalar: Unicode.Scalar) -> [Unicode.Scalar] {
        if scalar.value == 0x3000 { return [" "] }
        guard (0xFF01 ..< 0xFFF0).contains(scalar.value) else { return [scalar] }
        return Array(String(Character(scalar)).precomposedStringWithCompatibilityMapping.unicodeScalars)
    }

    /// `uncurl_quotes`: ʼ and ‘ ’ ‚ ‛ to ', “ ” „ ‟ to ".
    static func quote(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
        switch scalar.value {
        case 0x2BC, 0x2018 ... 0x201B: "'"
        case 0x201C ... 0x201F: "\""
        default: scalar
        }
    }

    /// `fix_line_breaks`: CRLF, CR, U+2028, U+2029 and U+85 to "\n".
    static func fixLineBreaks(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var index = 0
        while index < text.count {
            switch text[index].value {
            case 0x0D:
                result.append("\n")
                if index + 1 < text.count, text[index + 1] == "\n" { index += 1 }
            case 0x2028, 0x2029, 0x85: result.append("\n")
            default: result.append(text[index])
            }
            index += 1
        }
        return result
    }

    /// `remove_terminal_escapes`: ESC [ digits-or-semicolons letter.
    static func removeTerminalEscapes(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var index = 0
        while index < text.count {
            if text[index] == "\u{1B}", index + 1 < text.count, text[index + 1] == "[" {
                var end = index + 2
                while end < text.count, text[end] == ";" || text[end].properties.generalCategory == .decimalNumber { end += 1 }
                if end < text.count, text[end].isASCII, text[end].properties.isAlphabetic {
                    index = end + 1
                    continue
                }
            }
            result.append(text[index])
            index += 1
        }
        return result
    }

    /// `CONTROL_CHARS`: the controls `remove_control_chars` deletes.
    static func isControl(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x00 ... 0x08, 0x0B, 0x0E ... 0x1F, 0x7F, 0x206A ... 0x206F, 0xFEFF, 0xFFF9 ... 0xFFFC: true
        default: false
        }
    }

    /// `ftfy.fixes.unescape_html`: `&#?[0-9A-Za-z]{1,24};` replaced when it names an entity (with its semicolon) or
    /// is a reference by number that `html.unescape` decodes whole.
    static func ftfyUnescape(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var index = 0
        while index < text.count {
            guard text[index] == "&" else {
                result.append(text[index])
                index += 1
                continue
            }
            var end = index + 1
            if end < text.count, text[end] == "#" { end += 1 }
            let bodyStart = end
            while end < text.count, end - bodyStart < 24, text[end].isASCII, text[end].properties.isAlphabetic || ("0" ... "9").contains(text[end]) {
                end += 1
            }
            guard end > bodyStart, end < text.count, text[end] == ";" else {
                result.append(text[index])
                index += 1
                continue
            }
            let match = Array(text[index ... end])
            let name = PythonText.string(Array(match.dropFirst()))
            if match[1] != "#", let replacement = ftfyEntities[name] {
                result += replacement.unicodeScalars
            } else if match[1] == "#" {
                let decoded = htmlUnescape(match)
                result += decoded.contains(";") ? match : decoded
            } else {
                result += match
            }
            index = end + 1
        }
        return result
    }

    /// The entities ftfy decodes (with semicolons), its upper-case forms included, of the names this port knows.
    static let ftfyEntities: [String: String] = [
        "amp;": "&", "AMP;": "&", "lt;": "<", "LT;": "<", "gt;": ">", "GT;": ">", "quot;": "\"", "QUOT;": "\"",
        "apos;": "'", "APOS;": "'", "nbsp;": "\u{A0}", "NBSP;": "\u{A0}",
    ]

    /// The names `html.unescape` knows among those, with the legacy forms it also reads without a semicolon.
    static let htmlEntities: [String: String] = [
        "amp;": "&", "amp": "&", "AMP;": "&", "AMP": "&", "lt;": "<", "lt": "<", "LT;": "<", "LT": "<",
        "gt;": ">", "gt": ">", "GT;": ">", "GT": ">", "quot;": "\"", "quot": "\"", "QUOT;": "\"", "QUOT": "\"",
        "apos;": "'", "nbsp;": "\u{A0}", "nbsp": "\u{A0}",
    ]

    /// `html.unescape`: `&#digits;`, `&#xhex;` (the semicolon optional) and named references, the longest known
    /// name at the start of a name that is not one.
    static func htmlUnescape(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var index = 0
        func isDigit(_ s: Unicode.Scalar) -> Bool { ("0" ... "9").contains(s) }
        func isHex(_ s: Unicode.Scalar) -> Bool { isDigit(s) || ("a" ... "f").contains(s) || ("A" ... "F").contains(s) }
        while index < text.count {
            guard text[index] == "&", index + 1 < text.count else {
                result.append(text[index])
                index += 1
                continue
            }
            if text[index + 1] == "#" {
                var end = index + 2
                let hex = end < text.count && (text[end] == "x" || text[end] == "X")
                if hex { end += 1 }
                let digitsStart = end
                while end < text.count, hex ? isHex(text[end]) : isDigit(text[end]) { end += 1 }
                guard end > digitsStart else {
                    result.append(text[index])
                    index += 1
                    continue
                }
                let digits = PythonText.string(Array(text[digitsStart ..< end]))
                if end < text.count, text[end] == ";" { end += 1 }
                result += numericReference(digits, hex: hex)
                index = end
                continue
            }
            var end = index + 1
            while end < text.count, end - index - 1 < 32, !"\t\n\u{0C} <&#;".unicodeScalars.contains(text[end]) { end += 1 }
            guard end > index + 1 else {
                result.append(text[index])
                index += 1
                continue
            }
            if end < text.count, text[end] == ";" { end += 1 }
            let name = Array(text[index + 1 ..< end])
            if let replacement = htmlEntities[PythonText.string(name)] {
                result += replacement.unicodeScalars
            } else if let length = stride(from: name.count - 1, to: 1, by: -1).first(where: { htmlEntities[PythonText.string(Array(name[..<$0]))] != nil }) {
                result += htmlEntities[PythonText.string(Array(name[..<length]))]!.unicodeScalars
                result += name[length...]
            } else {
                result.append("&")
                result += name
            }
            index = end
        }
        return result
    }

    /// A reference by number as `html.unescape` decodes it: the Windows-1252 reading of U+80–U+9F (and U+0 → U+FFFD,
    /// U+D → "\r"), surrogates and numbers past U+10FFFF → U+FFFD, the code points HTML forbids → nothing.
    static func numericReference(_ digits: String, hex: Bool) -> [Unicode.Scalar] {
        let trimmed = digits.drop(while: { $0 == "0" })
        guard trimmed.count <= (hex ? 6 : 7), let number = UInt32(trimmed.isEmpty ? "0" : String(trimmed), radix: hex ? 16 : 10) else {
            return ["\u{FFFD}"]
        }
        switch number {
        case 0x00: return ["\u{FFFD}"]
        case 0x0D: return ["\r"]
        case 0x80 ... 0x9F: return [windows1252[Int(number - 0x80)]]
        case 0xD800 ... 0xDFFF, 0x110000...: return ["\u{FFFD}"]
        case 0x01 ... 0x08, 0x0B, 0x0E ... 0x1F, 0x7F, 0xFDD0 ... 0xFDEF: return []
        default:
            if number & 0xFFFE == 0xFFFE { return [] }
            return [Unicode.Scalar(number)!]
        }
    }
}
