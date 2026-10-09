import Foundation
import Observation

/// The LoRA files the user added, kept in the app's own folder: the engine, a child process in the
/// app's sandbox, can read that folder, but not a file picked in an open panel once it is running
/// (the access an open panel grants is the app's, and arrives too late to be inherited). A copy
/// on the same volume is a clone and takes no space.
@Observable
final class LoRALibrary {
    struct Entry: Identifiable, Hashable {
        /// The file's name in the folder.
        let file: String
        let bytes: Int64
        /// Nil when the header could not be read.
        let info: LoRAFileInfo?

        var id: String { file }
        var name: String { (file as NSString).deletingPathExtension }
    }

    enum ImportError: LocalizedError {
        case notSafetensors

        var errorDescription: String? {
            switch self {
            case .notSafetensors: "Choose a LoRA saved as .safetensors."
            }
        }
    }

    nonisolated static let directory = BackendController.supportDirectory.appending(path: "LoRAs", directoryHint: .isDirectory)

    nonisolated static func url(_ file: String) -> URL {
        directory.appending(path: file)
    }

    /// Every file of the folder, by name.
    private(set) var entries: [Entry] = []
    /// Headers already read, by file name, size and date.
    @ObservationIgnored private var infos: [String: (bytes: Int64, date: Date, info: LoRAFileInfo?)] = [:]

    func entry(_ file: String) -> Entry? {
        entries.first { $0.file == file }
    }

    /// Lists the folder again (files may have been added or removed in Finder).
    func refresh() {
        let fileManager = FileManager.default
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        let urls = (try? fileManager.contentsOfDirectory(at: Self.directory, includingPropertiesForKeys: keys)) ?? []
        var found: [Entry] = []
        for url in urls where url.pathExtension.lowercased() == "safetensors" && !url.lastPathComponent.hasPrefix(".") {
            let values = try? url.resourceValues(forKeys: Set(keys))
            let bytes = Int64(values?.fileSize ?? 0)
            let date = values?.contentModificationDate ?? .distantPast
            let file = url.lastPathComponent
            let info: LoRAFileInfo?
            if let known = infos[file], known.bytes == bytes, known.date == date {
                info = known.info
            } else {
                info = try? LoRAFileInfo.read(url)
                infos[file] = (bytes, date, info)
            }
            found.append(Entry(file: file, bytes: bytes, info: info))
        }
        let sorted = found.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        if sorted != entries { entries = sorted }
    }

    /// Copies a LoRA into the folder (the same file again is not copied twice) and returns its entry.
    func add(_ source: URL) async throws -> Entry {
        guard source.pathExtension.lowercased() == "safetensors" else { throw ImportError.notSafetensors }
        let info = try LoRAFileInfo.read(source)
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        let bytes = (try? source.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) } ?? 0
        let stem = source.deletingPathExtension().lastPathComponent
        var file = "\(stem).safetensors"
        var number = 2
        while fileManager.fileExists(atPath: Self.url(file).path) {
            if Self.isSame(Self.url(file), as: source, bytes: bytes) {
                refresh()
                return entry(file) ?? Entry(file: file, bytes: bytes, info: info)
            }
            file = "\(stem) \(number).safetensors"
            number += 1
        }
        let destination = Self.url(file)
        // A clone on the same volume, a real copy (hundreds of MB) from another one.
        try await Task.detached { try FileManager.default.copyItem(at: source, to: destination) }.value
        refresh()
        return entry(file) ?? Entry(file: file, bytes: bytes, info: info)
    }

    /// The file already in the folder: as large, with the same tensors and metadata.
    private nonisolated static func isSame(_ file: URL, as source: URL, bytes: Int64) -> Bool {
        guard (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize).map({ Int64($0) }) == bytes,
              let kept = try? LoRAFileInfo.header(of: file), let added = try? LoRAFileInfo.header(of: source)
        else { return false }
        return kept.tensors == added.tensors && kept.metadata == added.metadata
    }

    /// Moves a LoRA to the Trash.
    func remove(_ file: String) throws {
        try FileManager.default.trashItem(at: Self.url(file), resultingItemURL: nil)
        refresh()
    }
}
