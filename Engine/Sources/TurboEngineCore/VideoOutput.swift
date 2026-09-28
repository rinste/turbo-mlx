import AVFoundation
import CoreVideo
import Foundation
import MLX

/// MP4 files from decoded frames and a stereo waveform: H.264 video and AAC audio written with
/// `AVAssetWriter`, frames appended as they are decoded. The sound is known before the first frame
/// and is written about half a second ahead of the video: the writer interleaves its tracks and
/// stops taking frames when the other track falls behind. The sound ends a few hundredths of a
/// second before the last frame, so its track is closed as soon as all of it is written; left
/// open, the writer would wait for more near the end of the clip and never take the last frames.
public final class VideoOutput {
    public enum OutputError: LocalizedError {
        case cannotStart(String)
        case cannotWrite(String)

        public var errorDescription: String? {
            switch self {
            case .cannotStart(let reason): "Could not start writing the video: \(reason)"
            case .cannotWrite(let reason): "Could not write the video: \(reason)"
            }
        }
    }

    /// How far the sound runs ahead of the frames, and how much of it one sample buffer holds.
    private static let audioLead = 0.5
    private static let audioChunkSeconds = 0.25

    public let url: URL
    public let width: Int
    public let height: Int
    private let fps: Double
    private let writer: AVAssetWriter
    private let videoInput: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private var audioInput: AVAssetWriterInput?
    /// Interleaved stereo float32, and how many sample frames of it are written.
    private var audioSamples: [Float] = []
    private var audioWritten = 0
    private var audioFinished = false
    private let sampleRate: Int
    private var audioFormat: CMAudioFormatDescription?
    private let timescale: CMTimeScale = 6000
    public private(set) var frameCount = 0
    /// The first frame, [H, W, 3] uint8, for the poster.
    public private(set) var firstFrame: MLXArray?

    /// `audio`: [2, samples] float32 in [-1, 1] at `sampleRate`, or nil for a silent clip.
    public init(url: URL, width: Int, height: Int, fps: Double, audio: MLXArray?, sampleRate: Int = 48000) throws {
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.sampleRate = sampleRate
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        do {
            writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            throw OutputError.cannotStart(error.localizedDescription)
        }
        // High quality: about 0.3 bits per pixel per frame at 24 fps, never below 8 Mbit/s.
        let bitrate = max(8_000_000, Int(Double(width * height) * fps * 0.3))
        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoExpectedSourceFrameRateKey: fps,
            ],
        ])
        videoInput.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
        guard writer.canAdd(videoInput) else { throw OutputError.cannotStart("the video track was refused") }
        writer.add(videoInput)
        if let audio, audio.shape[1] > 0 {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 256_000,
            ])
            input.expectsMediaDataInRealTime = false
            if writer.canAdd(input) {
                writer.add(input)
                audioInput = input
                let interleaved = audio.transposed(1, 0).asType(.float32)
                eval(interleaved)
                audioSamples = interleaved.asArray(Float.self)
                var description = AudioStreamBasicDescription(
                    mSampleRate: Float64(sampleRate), mFormatID: kAudioFormatLinearPCM,
                    mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                    mBytesPerPacket: 8, mFramesPerPacket: 1, mBytesPerFrame: 8, mChannelsPerFrame: 2, mBitsPerChannel: 32,
                    mReserved: 0
                )
                CMAudioFormatDescriptionCreate(allocator: nil, asbd: &description, layoutSize: 0, layout: nil,
                                               magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &audioFormat)
            }
        }
        guard writer.startWriting() else {
            throw OutputError.cannotStart(writer.error?.localizedDescription ?? "unknown error")
        }
        writer.startSession(atSourceTime: .zero)
    }

    private func time(ofFrame index: Int) -> CMTime {
        CMTime(value: CMTimeValue((Double(index) * Double(timescale) / fps).rounded()), timescale: timescale)
    }

    /// Appends frames [N, H, W, 3] uint8.
    public func append(frames: MLXArray) throws {
        let count = frames.shape[0]
        let bgra = concatenated([frames[.ellipsis, 2 ..< 3], frames[.ellipsis, 1 ..< 2], frames[.ellipsis, 0 ..< 1],
                                 MLXArray.full([count, height, width, 1], values: MLXArray(UInt8(255)))], axis: -1)
        eval(bgra)
        if firstFrame == nil { firstFrame = frames[0] }
        let bytes = bgra.asData(noCopy: false)
        let frameBytes = width * height * 4
        for index in 0 ..< count {
            // The sound a little ahead of this frame, then the frame once the writer takes it.
            try feedAudio(upTo: Int((Double(frameCount + 1) / fps + Self.audioLead) * Double(sampleRate)))
            let started = Date()
            while !videoInput.isReadyForMoreMediaData {
                if try !feedAudio(upTo: audioSamples.count / 2) { Thread.sleep(forTimeInterval: 0.002) }
                if Date().timeIntervalSince(started) > 60 {
                    throw OutputError.cannotWrite(writer.error?.localizedDescription ?? "the writer stopped taking frames")
                }
            }
            guard let pool = adaptor.pixelBufferPool else { throw OutputError.cannotWrite("no pixel buffer pool") }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { throw OutputError.cannotWrite("no pixel buffer") }
            CVPixelBufferLockBaseAddress(buffer, [])
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                bytes.withUnsafeBytes { source in
                    for row in 0 ..< height {
                        memcpy(base + row * rowBytes, source.baseAddress! + index * frameBytes + row * width * 4, width * 4)
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            guard adaptor.append(buffer, withPresentationTime: time(ofFrame: frameCount)) else {
                throw OutputError.cannotWrite(writer.error?.localizedDescription ?? "frame \(frameCount) was refused")
            }
            frameCount += 1
        }
    }

    /// Writes the rest of the sound, cut to the video's length, and closes the file.
    public func finish() throws {
        videoInput.markAsFinished()
        if let audioInput, !audioFinished {
            let end = min(audioSamples.count / 2, Int((Double(frameCount) / fps * Double(sampleRate)).rounded()))
            let started = Date()
            while audioWritten < end, !audioFinished {
                if try !feedAudio(upTo: end) { Thread.sleep(forTimeInterval: 0.002) }
                if Date().timeIntervalSince(started) > 60 {
                    throw OutputError.cannotWrite(writer.error?.localizedDescription ?? "the writer stopped taking sound")
                }
            }
            if !audioFinished { audioInput.markAsFinished() }
        }
        writer.endSession(atSourceTime: time(ofFrame: frameCount))
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        if writer.status != .completed {
            throw OutputError.cannotWrite(writer.error?.localizedDescription ?? "status \(writer.status.rawValue)")
        }
    }

    /// Stops writing and deletes the file.
    public func cancel() {
        if writer.status == .writing { writer.cancelWriting() }
        try? FileManager.default.removeItem(at: url)
    }

    /// Appends sound up to sample frame `limit` while the audio input takes it, a quarter of a
    /// second per buffer, and closes the track after the last of it. Returns whether it appended
    /// anything.
    @discardableResult
    private func feedAudio(upTo limit: Int) throws -> Bool {
        guard let audioInput, let audioFormat, !audioFinished else { return false }
        let target = min(limit, audioSamples.count / 2)
        var appended = false
        while audioWritten < target, audioInput.isReadyForMoreMediaData {
            let count = min(Int(Double(sampleRate) * Self.audioChunkSeconds), target - audioWritten)
            let byteCount = count * 8
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: byteCount, blockAllocator: nil,
                                               customBlockSource: nil, offsetToData: 0, dataLength: byteCount, flags: 0,
                                               blockBufferOut: &block)
            guard let block else { throw OutputError.cannotWrite("no audio buffer") }
            let offset = audioWritten
            audioSamples.withUnsafeBytes { source in
                _ = CMBlockBufferReplaceDataBytes(with: source.baseAddress! + offset * 8, blockBuffer: block,
                                                  offsetIntoDestination: 0, dataLength: byteCount)
            }
            var sample: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                allocator: nil, dataBuffer: block, formatDescription: audioFormat, sampleCount: count,
                presentationTimeStamp: CMTime(value: CMTimeValue(offset), timescale: CMTimeScale(sampleRate)),
                packetDescriptions: nil, sampleBufferOut: &sample
            )
            guard let sample, audioInput.append(sample) else {
                throw OutputError.cannotWrite(writer.error?.localizedDescription ?? "the sound was refused")
            }
            audioWritten += count
            appended = true
        }
        if audioWritten == audioSamples.count / 2 {
            audioInput.markAsFinished()
            audioFinished = true
        }
        return appended
    }
}
