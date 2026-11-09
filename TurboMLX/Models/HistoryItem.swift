import Foundation
import UniformTypeIdentifiers

/// What a model produces, and what a history item holds.
nonisolated enum MediaKind: String, Codable, Hashable, Sendable {
    case image
    case video
    /// A 3D object (Idol): a GLB, with a PNG still the app draws of it.
    case model3D = "model3d"
    /// A song (MiniMax-Music3) or a sound effect (MOSS-SoundEffect): a WAV, with a PNG of its waveform the app draws.
    case audio

    /// What menus and tooltips call it: "image", "video", "3D object", "song".
    var noun: String {
        switch self {
        case .image: "image"
        case .video: "video"
        case .model3D: "3D object"
        case .audio: "song"
        }
    }

    /// "Image", "Video", "3D Object", "Song".
    var title: String {
        switch self {
        case .image: "Image"
        case .video: "Video"
        case .model3D: "3D Object"
        case .audio: "Song"
        }
    }
}

/// A finished image (or clip) and the settings that produced it.
nonisolated struct HistoryItem: Identifiable, Hashable, Codable, Sendable {
    var id: UUID
    var createdAt: Date
    /// File name inside the history folder: a PNG, a movie for `.video`, a GLB for `.model3D`, a WAV for `.audio`.
    var fileName: String
    /// Absent in items saved before there were videos: an image.
    var media: MediaKind?
    /// A PNG frame of a video, a still of a 3D object or a song's waveform, for thumbnails and Quick Look.
    var posterFileName: String?
    var modelID: String
    var modelName: String
    var request: GenerationRequest
    var seconds: Double
    var peakMemory: Int64?
    /// Seconds per phase (load, encode, denoise, decode, save); older items have none.
    var timings: [String: Double]?
    /// True once starred; absent otherwise, as in items saved before there were favorites, so the index
    /// reads as before and older versions, which skip a field they do not know, still open it.
    var favorite: Bool?

    var prompt: String { request.prompt }
    /// The prompt, or what an upscale did.
    var caption: String { request.caption }
    var size: PixelSize { request.size }
    var kind: MediaKind { media ?? .image }
    /// A sound without lyrics: a sound effect (MOSS-SoundEffect), not a song.
    var isSoundEffect: Bool { kind == .audio && request.lyrics == nil }
    /// What menus and tooltips call it: the kind's name, "sound" for a sound effect.
    var noun: String { isSoundEffect ? "sound" : kind.noun }
    var title: String { isSoundEffect ? "Sound" : kind.title }
    var isFavorite: Bool { favorite == true }
}

/// How the strip arranges the history, left to right: by date, oldest first, or in the order the
/// user dragged the items into (the newer ones after it).
nonisolated enum HistoryOrder: String, CaseIterable, Identifiable, Sendable {
    case date
    case custom

    var id: Self { self }

    var title: String {
        switch self {
        case .date: "By Date"
        case .custom: "Custom Order"
        }
    }
}

/// What the strip shows of the history: with a search, the items that have each of its words in their
/// prompt, negative prompt, lyrics, LoRAs or model name, whatever the case and the accents; with
/// `favoritesOnly`, the starred ones. Neither is saved: a search is for the moment, a star for good,
/// and a strip that opened on part of the history would hide the newest generations.
nonisolated struct HistoryFilter: Equatable, Sendable {
    var search = ""
    var favoritesOnly = false

    var isActive: Bool { favoritesOnly || !words.isEmpty }

    private var words: [Substring] { search.split(whereSeparator: \.isWhitespace) }

    func admits(_ item: HistoryItem) -> Bool {
        if favoritesOnly, !item.isFavorite { return false }
        let words = words
        guard !words.isEmpty else { return true }
        let request = item.request
        let texts = [request.prompt, request.negativePrompt, request.lyrics, item.modelName].compactMap { $0 }
            + (request.loras ?? []).map(\.name)
        return words.allSatisfy { word in texts.contains { $0.localizedStandardContains(word) } }
    }
}

extension UTType {
    /// A history item dragged inside the strip, to move it (declared in TurboMLX-Info.plist). The
    /// image or clip goes along as a file for other apps and the reference image.
    static let historyItem = UTType(exportedAs: "io.github.rinste.TurboMLX.history-item")
}
