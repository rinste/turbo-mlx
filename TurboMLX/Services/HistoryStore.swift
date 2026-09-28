import AVFoundation
import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Observation

/// Generated images and their settings, stored as PNGs plus a JSON index in Application Support.
@Observable
final class HistoryStore {
    static let directory = BackendController.supportDirectory.appending(path: "History", directoryHint: .isDirectory)

    /// Newest first.
    private(set) var items: [HistoryItem] = []
    private(set) var loadError: String?

    @ObservationIgnored private var reservedNames: Set<String> = []
    private var indexURL: URL { Self.directory.appending(path: "history.json") }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    private static let fileStamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()

    func load() {
        do {
            try FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
            guard FileManager.default.fileExists(atPath: indexURL.path) else { return }
            let stored = try Self.decoder.decode([HistoryItem].self, from: Data(contentsOf: indexURL))
            // Images deleted from Finder simply drop out of the history.
            items = stored.filter { FileManager.default.fileExists(atPath: url(for: $0).path) }
        } catch {
            loadError = error.localizedDescription
        }
    }

    func item(withID id: HistoryItem.ID?) -> HistoryItem? {
        guard let id else { return nil }
        return items.first { $0.id == id }
    }

    func url(for item: HistoryItem) -> URL {
        Self.directory.appending(path: item.fileName)
    }

    /// The still to show for an item: the image itself, or a video's poster frame.
    func posterURL(for item: HistoryItem) -> URL {
        Self.directory.appending(path: item.posterFileName ?? item.fileName)
    }

    /// A fresh file name; queued jobs have not written theirs yet, so names handed out are remembered.
    func newImageURL(seed: Int) -> URL { newURL(seed: seed, extension: "png") }

    /// A clip's file; the engine writes its poster next to it, with the same name as a PNG.
    func newVideoURL(seed: Int) -> URL { newURL(seed: seed, extension: "mp4") }

    private func newURL(seed: Int, extension ext: String) -> URL {
        let stamp = Self.fileStamp.string(from: Date())
        var name = "\(stamp)-\(seed).\(ext)"
        if reservedNames.contains(name) || FileManager.default.fileExists(atPath: Self.directory.appending(path: name).path) {
            name = "\(stamp)-\(seed)-\(UUID().uuidString.prefix(6)).\(ext)"
        }
        reservedNames.insert(name)
        return Self.directory.appending(path: name)
    }

    // MARK: Reference images

    /// Images clips start from, copied here so they outlive the file they came from (and so the
    /// engine, inside the app's sandbox, can read them).
    nonisolated static let referencesDirectory = BackendController.supportDirectory.appending(path: "References", directoryHint: .isDirectory)

    nonisolated static func referenceURL(_ name: String) -> URL {
        referencesDirectory.appending(path: name)
    }

    /// Copies an image into the references folder as a PNG (at most 2048 pixels on the long
    /// side) and returns its name. A clip stands for its first frame: the poster next to it when it
    /// is one of ours, read from the video otherwise.
    static func importReference(from url: URL) async throws -> String {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        if UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) == true {
            let poster = url.deletingPathExtension().appendingPathExtension("png")
            if FileManager.default.isReadableFile(atPath: poster.path),
               let imageSource = CGImageSourceCreateWithURL(poster as CFURL, nil) {
                return try importReference(imageSource)
            }
            return try importReference(image: await firstFrame(of: url))
        }
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else { throw ReferenceError.unreadable }
        return try importReference(imageSource)
    }

    /// The same, for image data (dragged from a browser or Photos).
    static func importReference(data: Data) throws -> String {
        guard let imageSource = CGImageSourceCreateWithData(data as CFData, nil) else { throw ReferenceError.unreadable }
        return try importReference(imageSource)
    }

    private static func importReference(image: CGImage) throws -> String {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw ReferenceError.unreadable }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ReferenceError.unreadable }
        return try importReference(data: data as Data)
    }

    private static func firstFrame(of url: URL) async throws -> CGImage {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        do {
            return try await generator.image(at: .zero).image
        } catch {
            throw ReferenceError.unreadable
        }
    }

    private static func importReference(_ imageSource: CGImageSource) throws -> String {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 2048,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, options as CFDictionary) else {
            throw ReferenceError.unreadable
        }
        try FileManager.default.createDirectory(at: referencesDirectory, withIntermediateDirectories: true)
        let name = "\(UUID().uuidString).png"
        guard let destination = CGImageDestinationCreateWithURL(referenceURL(name) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ReferenceError.unreadable }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ReferenceError.unreadable }
        return name
    }

    /// Deletes the reference images that no clip in the history and not `current` start from.
    func pruneReferences(keeping current: String?) {
        guard loadError == nil else { return }
        let used = Set(items.compactMap(\.request.referenceImage) + [current].compactMap { $0 })
        let names = (try? FileManager.default.contentsOfDirectory(atPath: Self.referencesDirectory.path)) ?? []
        for name in names where !used.contains(name) {
            try? FileManager.default.removeItem(at: Self.referenceURL(name))
        }
    }

    enum ReferenceError: LocalizedError {
        case unreadable
        var errorDescription: String? { "This file is not an image or a clip Turbo MLX can read." }
    }

    func add(_ item: HistoryItem) {
        items.insert(item, at: 0)
        save()
    }

    /// Moves the images to the Trash, so a deletion can still be undone from Finder.
    func remove(_ ids: Set<HistoryItem.ID>) {
        for item in items where ids.contains(item.id) {
            try? FileManager.default.trashItem(at: url(for: item), resultingItemURL: nil)
            if item.posterFileName != nil { try? FileManager.default.trashItem(at: posterURL(for: item), resultingItemURL: nil) }
            ImageLoader.evict(posterURL(for: item))
        }
        items.removeAll { ids.contains($0.id) }
        save()
    }

    func removeAll() {
        remove(Set(items.map(\.id)))
    }

    private func save() {
        do {
            try FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
            try Self.encoder.encode(items).write(to: indexURL, options: .atomic)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

/// Decodes images off the main thread and keeps recent ones in memory.
enum ImageLoader {
    private static let cache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 300
        return cache
    }()

    /// `maxPixelSize` nil loads the full image.
    static func cached(_ url: URL, maxPixelSize: Int?) -> NSImage? {
        cache.object(forKey: key(url, maxPixelSize))
    }

    static func load(_ url: URL, maxPixelSize: Int?) async -> NSImage? {
        if let hit = cached(url, maxPixelSize: maxPixelSize) { return hit }
        guard let cgImage = await decode(url, maxPixelSize: maxPixelSize) else { return nil }
        let image = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        cache.setObject(image, forKey: key(url, maxPixelSize))
        return image
    }

    static func evict(_ url: URL) {
        // Keys embed the size, so drop the common ones.
        for size in [nil, 160, 320, 512] as [Int?] {
            cache.removeObject(forKey: key(url, size))
        }
    }

    private static func key(_ url: URL, _ maxPixelSize: Int?) -> NSString {
        "\(url.path)#\(maxPixelSize.map(String.init) ?? "full")" as NSString
    }

    @concurrent
    nonisolated private static func decode(_ url: URL, maxPixelSize: Int?) async -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        guard let maxPixelSize else {
            return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }
}
