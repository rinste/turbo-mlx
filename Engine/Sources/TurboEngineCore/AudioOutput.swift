import Foundation
import MLX

/// WAV files from a generated waveform: 16-bit PCM, the generation's parameters as JSON in the INFO comment (as a PNG
/// keeps them in its description) with the tool in ISFT, and the mark of generative AI in XMP, in the `_PMX` chunk
/// where Adobe's specification and exiftool put it in a RIFF file (`Provenance`).
public enum AudioOutput {
    public enum OutputError: LocalizedError {
        case unsupportedShape([Int])

        public var errorDescription: String? {
            switch self {
            case .unsupportedShape(let shape): "Cannot make a sound file from an array of shape \(shape)."
            }
        }
    }

    /// `samples` is [frames, channels] float in −1…1.
    public static func writeWAV(_ samples: MLXArray, sampleRate: Int, to url: URL, source: Provenance.SourceType,
                                metadata: [String: Any]) throws {
        guard samples.ndim == 2, (1 ... 2).contains(samples.dim(1)) else { throw OutputError.unsupportedShape(samples.shape) }
        let channels = samples.dim(1)
        let pcm = (clip(samples.asType(.float32), min: -1, max: 1) * 32767).round().asType(.int16)
        let data = pcm.asData(access: .copy).data

        var info = Data()
        info.append(contentsOf: Array("INFO".utf8))
        info.append(chunk("ISFT", Data((Provenance.software + "\0").utf8)))
        if let json = try? JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]) {
            info.append(chunk("ICMT", json + Data([0])))
        }

        var body = Data()
        body.append(contentsOf: Array("WAVE".utf8))
        var format = Data()
        format.append(le(UInt16(1)))
        format.append(le(UInt16(channels)))
        format.append(le(UInt32(sampleRate)))
        format.append(le(UInt32(sampleRate * channels * 2)))
        format.append(le(UInt16(channels * 2)))
        format.append(le(UInt16(16)))
        body.append(chunk("fmt ", format))
        body.append(chunk("LIST", info))
        body.append(chunk("_PMX", Data(Provenance.xmpPacket(source).utf8)))
        body.append(chunk("data", data))

        var file = Data()
        file.append(contentsOf: Array("RIFF".utf8))
        file.append(le(UInt32(body.count)))
        file.append(body)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try file.write(to: url, options: .atomic)
    }

    /// A RIFF chunk, padded to an even length.
    static func chunk(_ id: String, _ payload: Data) -> Data {
        var data = Data(id.utf8)
        data.append(le(UInt32(payload.count)))
        data.append(payload)
        if payload.count % 2 == 1 { data.append(0) }
        return data
    }

    static func le<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }

    /// The chunks of a RIFF/WAVE file by id (the first of each), or nil if it is not one.
    public static func chunks(ofWAV url: URL) -> [String: Data]? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe), data.count >= 12,
              data[0 ..< 4] == Data("RIFF".utf8), data[8 ..< 12] == Data("WAVE".utf8) else { return nil }
        var chunks: [String: Data] = [:]
        var offset = 12
        while offset + 8 <= data.count {
            let id = String(decoding: data[offset ..< offset + 4], as: UTF8.self)
            let size = data[offset + 4 ..< offset + 8].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
            guard offset + 8 + size <= data.count else { break }
            if chunks[id] == nil { chunks[id] = data.subdata(in: offset + 8 ..< offset + 8 + size) }
            offset += 8 + size + size % 2
        }
        return chunks
    }
}
