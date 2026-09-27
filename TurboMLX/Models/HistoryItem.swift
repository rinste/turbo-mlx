import Foundation

/// A finished image and the settings that produced it.
nonisolated struct HistoryItem: Identifiable, Hashable, Codable, Sendable {
    var id: UUID
    var createdAt: Date
    /// File name inside the history folder.
    var fileName: String
    var modelID: String
    var modelName: String
    var request: GenerationRequest
    var seconds: Double
    var peakMemory: Int64?

    var prompt: String { request.prompt }
    var size: PixelSize { request.size }
}
