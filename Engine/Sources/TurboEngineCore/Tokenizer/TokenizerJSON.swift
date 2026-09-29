import Foundation

/// A JSON value read byte for byte, for `tokenizer.json`. Foundation's parsers do not fit it:
/// `JSONSerialization` drops a U+FEFF at the start of a string (Gemma's vocabulary has such
/// tokens), and anything keyed by Swift `String` merges canonically equivalent keys (";" U+003B and
/// U+037E, a combining grave U+0300 and U+0340), which a vocabulary keeps apart. Objects keep their
/// members as a list, strings their exact scalars.
enum TokenizerJSON {
    case object([(key: String, value: TokenizerJSON)])
    case array([TokenizerJSON])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    struct ParseError: Error {
        let offset: Int
    }

    static func parse(_ data: Data) throws -> TokenizerJSON {
        try data.withUnsafeBytes { raw in
            var parser = Parser(bytes: raw.bindMemory(to: UInt8.self))
            let value = try parser.value()
            parser.skipWhitespace()
            guard parser.position == parser.bytes.count else { throw ParseError(offset: parser.position) }
            return value
        }
    }

    subscript(key: String) -> TokenizerJSON? {
        guard case .object(let members) = self else { return nil }
        return members.first { $0.key.unicodeScalars.elementsEqual(key.unicodeScalars) }?.value
    }

    var string: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var int: Int? {
        if case .number(let value) = self { return Int(value) }
        return nil
    }

    var bool: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    var array: [TokenizerJSON]? {
        if case .array(let values) = self { return values }
        return nil
    }

    var members: [(key: String, value: TokenizerJSON)]? {
        if case .object(let members) = self { return members }
        return nil
    }

    private struct Parser {
        let bytes: UnsafeBufferPointer<UInt8>
        var position = 0

        mutating func skipWhitespace() {
            while position < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[position]) { position += 1 }
        }

        mutating func value() throws -> TokenizerJSON {
            skipWhitespace()
            guard position < bytes.count else { throw ParseError(offset: position) }
            switch bytes[position] {
            case UInt8(ascii: "{"): return try object()
            case UInt8(ascii: "["): return try array()
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try literal("true"); return .bool(true)
            case UInt8(ascii: "f"): try literal("false"); return .bool(false)
            case UInt8(ascii: "n"): try literal("null"); return .null
            default: return .number(try number())
            }
        }

        mutating func literal(_ word: String) throws {
            for byte in word.utf8 {
                guard position < bytes.count, bytes[position] == byte else { throw ParseError(offset: position) }
                position += 1
            }
        }

        mutating func object() throws -> TokenizerJSON {
            position += 1
            var members: [(key: String, value: TokenizerJSON)] = []
            skipWhitespace()
            if position < bytes.count, bytes[position] == UInt8(ascii: "}") { position += 1; return .object(members) }
            while true {
                skipWhitespace()
                guard position < bytes.count, bytes[position] == UInt8(ascii: "\"") else { throw ParseError(offset: position) }
                let key = try string()
                skipWhitespace()
                guard position < bytes.count, bytes[position] == UInt8(ascii: ":") else { throw ParseError(offset: position) }
                position += 1
                members.append((key, try value()))
                skipWhitespace()
                guard position < bytes.count else { throw ParseError(offset: position) }
                if bytes[position] == UInt8(ascii: ",") { position += 1; continue }
                if bytes[position] == UInt8(ascii: "}") { position += 1; return .object(members) }
                throw ParseError(offset: position)
            }
        }

        mutating func array() throws -> TokenizerJSON {
            position += 1
            var values: [TokenizerJSON] = []
            skipWhitespace()
            if position < bytes.count, bytes[position] == UInt8(ascii: "]") { position += 1; return .array(values) }
            while true {
                values.append(try value())
                skipWhitespace()
                guard position < bytes.count else { throw ParseError(offset: position) }
                if bytes[position] == UInt8(ascii: ",") { position += 1; continue }
                if bytes[position] == UInt8(ascii: "]") { position += 1; return .array(values) }
                throw ParseError(offset: position)
            }
        }

        mutating func number() throws -> Double {
            let start = position
            while position < bytes.count, "+-0123456789.eE".utf8.contains(bytes[position]) { position += 1 }
            guard position > start, let value = Double(String(decoding: UnsafeBufferPointer(rebasing: bytes[start ..< position]), as: UTF8.self))
            else { throw ParseError(offset: start) }
            return value
        }

        /// The string's exact scalars: UTF-8 copied as it is, escapes decoded (surrogate pairs too).
        mutating func string() throws -> String {
            position += 1
            var scalars = String.UnicodeScalarView()
            var run = position
            func flush(_ end: Int) {
                guard end > run else { return }
                scalars.append(contentsOf: String(decoding: UnsafeBufferPointer(rebasing: bytes[run ..< end]), as: UTF8.self).unicodeScalars)
            }
            while position < bytes.count {
                let byte = bytes[position]
                if byte == UInt8(ascii: "\"") {
                    flush(position)
                    position += 1
                    return String(scalars)
                }
                if byte == UInt8(ascii: "\\") {
                    flush(position)
                    position += 1
                    guard position < bytes.count else { break }
                    let escape = bytes[position]
                    position += 1
                    switch escape {
                    case UInt8(ascii: "\""): scalars.append("\"")
                    case UInt8(ascii: "\\"): scalars.append("\\")
                    case UInt8(ascii: "/"): scalars.append("/")
                    case UInt8(ascii: "b"): scalars.append("\u{08}")
                    case UInt8(ascii: "f"): scalars.append("\u{0C}")
                    case UInt8(ascii: "n"): scalars.append("\n")
                    case UInt8(ascii: "r"): scalars.append("\r")
                    case UInt8(ascii: "t"): scalars.append("\t")
                    case UInt8(ascii: "u"):
                        var code = try hex4()
                        if (0xD800 ... 0xDBFF).contains(code), position + 1 < bytes.count,
                           bytes[position] == UInt8(ascii: "\\"), bytes[position + 1] == UInt8(ascii: "u") {
                            position += 2
                            let low = try hex4()
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        }
                        scalars.append(Unicode.Scalar(code) ?? "\u{FFFD}")
                    default:
                        throw ParseError(offset: position - 1)
                    }
                    run = position
                    continue
                }
                position += 1
            }
            throw ParseError(offset: position)
        }

        mutating func hex4() throws -> UInt32 {
            guard position + 4 <= bytes.count,
                  let value = UInt32(String(decoding: UnsafeBufferPointer(rebasing: bytes[position ..< position + 4]), as: UTF8.self), radix: 16)
            else { throw ParseError(offset: position) }
            position += 4
            return value
        }
    }
}
