import Foundation
import ImageIO

/// Marks every image and clip the engine makes as made by generative AI, in a form machines read:
/// IPTC's Digital Source Type in XMP, the field Google, Adobe, Meta and exiftool read, which the EU
/// AI Act (Article 50(2)) asks generative systems to provide in some machine-readable form. A PNG
/// carries it in its XMP (ImageIO's iTXt chunk), an MP4 in a top-level XMP `uuid` box, where
/// Adobe's specification and exiftool put it.
public enum Provenance {
    /// IPTC's Digital Source Type of a result (cv.iptc.org/newscodes/digitalsourcetype).
    public enum SourceType: String, CaseIterable, Sendable {
        /// "Created using Generative AI": from a prompt.
        case trainedAlgorithmicMedia
        /// "Edited using Generative AI": a picture changed, animated or upscaled by a model.
        case compositeWithTrainedAlgorithmicMedia

        public var uri: String { "http://cv.iptc.org/newscodes/digitalsourcetype/" + rawValue }
    }

    static let software = "Turbo MLX"

    /// What a result made from `input` is: created by AI from a prompt when there is no input, or
    /// when the input was itself created by AI (it says so, as the engine's own images do);
    /// edited by AI otherwise.
    public static func sourceType(input: URL?) -> SourceType {
        guard let input else { return .trainedAlgorithmicMedia }
        return sourceType(of: input) == .trainedAlgorithmicMedia ? .trainedAlgorithmicMedia : .compositeWithTrainedAlgorithmicMedia
    }

    /// The Digital Source Type an image file declares in its XMP, if it declares one.
    public static func sourceType(of url: URL) -> SourceType? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let metadata = CGImageSourceCopyMetadataAtIndex(source, 0, nil),
              let value = CGImageMetadataCopyStringValueWithPath(metadata, nil, sourceTypePath) as String?
        else { return nil }
        return SourceType.allCases.first { $0.uri == value }
    }

    /// The XMP ImageIO writes into a PNG: the Digital Source Type and the tool that made it.
    static func imageMetadata(_ type: SourceType) -> CGImageMetadata {
        let metadata = CGImageMetadataCreateMutable()
        CGImageMetadataSetValueWithPath(metadata, nil, sourceTypePath, type.uri as CFString)
        CGImageMetadataSetValueWithPath(metadata, nil, "\(kCGImageMetadataPrefixXMPBasic):CreatorTool" as CFString, software as CFString)
        return metadata
    }

    /// Appends the XMP box to a finished MP4. A top-level box of its own after the others moves no
    /// offset in the file, and players skip a box they do not know.
    public static func markVideo(at url: URL, as type: SourceType) throws {
        let packet = """
            <?xpacket begin="\u{FEFF}" id="W5M0MpCehiHzreSzNTczkc9d"?>
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
             <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
              <rdf:Description rdf:about=""
                xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
                xmlns:xmp="http://ns.adobe.com/xap/1.0/"
               Iptc4xmpExt:DigitalSourceType="\(type.uri)"
               xmp:CreatorTool="\(software)"/>
             </rdf:RDF>
            </x:xmpmeta>
            <?xpacket end="w"?>
            """
        let payload = Data(packet.utf8)
        var box = Data()
        var size = UInt32(8 + xmpBoxUUID.count + payload.count).bigEndian
        withUnsafeBytes(of: &size) { box.append(contentsOf: $0) }
        box.append(contentsOf: Array("uuid".utf8))
        box.append(contentsOf: xmpBoxUUID)
        box.append(payload)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: box)
    }

    /// The Digital Source Type an MP4 declares in its XMP box, if it has one.
    public static func sourceType(ofVideo url: URL) -> SourceType? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        func number(_ range: Range<Int>) -> Int { data[range].reduce(0) { $0 << 8 | Int($1) } }
        var offset = 0
        while offset + 8 <= data.count {
            var size = number(offset ..< offset + 4)
            let kind = String(decoding: data[offset + 4 ..< offset + 8], as: UTF8.self)
            // 1: a 64-bit size follows (an mdat past 4 GB); 0: the box runs to the end of the file.
            if size == 1, offset + 16 <= data.count { size = number(offset + 8 ..< offset + 16) }
            if size == 0 { size = data.count - offset }
            guard size >= 8, offset + size <= data.count else { return nil }
            if kind == "uuid", size > 24, Array(data[offset + 8 ..< offset + 24]) == xmpBoxUUID {
                let text = String(decoding: data[offset + 24 ..< offset + size], as: UTF8.self)
                return SourceType.allCases.first { text.contains("\"\($0.uri)\"") }
            }
            offset += size
        }
        return nil
    }

    private static let sourceTypePath = "\(kCGImageMetadataPrefixIPTCExtension):DigitalSourceType" as CFString
    /// Adobe's XMP UUID, BE7ACFCB-97A9-42E8-9C71-999491E3AFAC, in file order.
    private static let xmpBoxUUID: [UInt8] = [0xBE, 0x7A, 0xCF, 0xCB, 0x97, 0xA9, 0x42, 0xE8, 0x9C, 0x71, 0x99, 0x94, 0x91, 0xE3, 0xAF, 0xAC]
}
