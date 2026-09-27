import Foundation

/// What a model produces, and what a history item holds.
nonisolated enum MediaKind: String, Codable, Hashable, Sendable {
    case image
    case video
}

/// A finished image (or clip) and the settings that produced it.
nonisolated struct HistoryItem: Identifiable, Hashable, Codable, Sendable {
    var id: UUID
    var createdAt: Date
    /// File name inside the history folder: a PNG, or a movie for `.video`.
    var fileName: String
    /// Absent in items saved before there were videos: an image.
    var media: MediaKind?
    /// A PNG frame of a video, for thumbnails and Quick Look.
    var posterFileName: String?
    var modelID: String
    var modelName: String
    var request: GenerationRequest
    var seconds: Double
    var peakMemory: Int64?
    /// Seconds per phase (load, encode, denoise, decode, save); older items have none.
    var timings: [String: Double]?

    var prompt: String { request.prompt }
    var size: PixelSize { request.size }
    var kind: MediaKind { media ?? .image }
}
