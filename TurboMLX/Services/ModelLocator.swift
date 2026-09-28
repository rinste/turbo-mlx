import Foundation

/// Finds model weights on disk: Hugging Face repos in the hub cache (where mflux looks for them)
/// and local folders.
nonisolated struct ModelLocator: Sendable {
    /// `~/.cache/huggingface`, the folder the huggingface CLI, mflux and other tools share. The
    /// app's sandbox reaches it through an entitlement for this one path (TurboMLX.entitlements),
    /// and the engine, which inherits the sandbox, reads the models there too.
    static let huggingFaceHome = realHome.appending(path: ".cache/huggingface", directoryHint: .isDirectory)

    /// The user's home folder: in the sandbox, FileManager's is the app's container.
    private static var realHome: URL {
        if let entry = getpwuid(getuid()), let directory = entry.pointee.pw_dir {
            return URL(fileURLWithPath: String(cString: directory), isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }

    let hubCache = ModelLocator.huggingFaceHome.appending(path: "hub", directoryHint: .isDirectory)

    /// The folder holding a complete copy of the model, or nil if it still has to be downloaded.
    func installedLocation(of model: ModelDescriptor) -> URL? {
        switch model.source {
        case .local(let path):
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            return Self.isComplete(url, family: model.family) ? url : nil
        case .huggingFace(let repo):
            let snapshots = repoFolder(repo).appending(path: "snapshots")
            let keys: [URLResourceKey] = [.contentModificationDateKey]
            guard let entries = try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: keys)
            else { return nil }
            let newestFirst = entries.sorted {
                let lhs = (try? $0.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
                let rhs = (try? $1.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
                return lhs > rhs
            }
            return newestFirst.first { Self.isComplete($0, family: model.family) }
        }
    }

    func repoFolder(_ repo: String) -> URL {
        hubCache.appending(path: "models--" + repo.replacingOccurrences(of: "/", with: "--"))
    }

    /// Mirrors mflux's check: every component folder has its safetensors, and every shard named
    /// by an index is present (a dangling cache symlink does not count). The tokenizer must be
    /// there too: an interrupted download can leave a snapshot with weights but no tokenizer.
    static func isComplete(_ folder: URL, family: ModelFamily) -> Bool {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: folder.appending(path: family.tokenizerFile).path) else { return false }
        return family.components.allSatisfy { component in
            let directory = folder.appending(path: component)
            guard let files = try? fileManager.contentsOfDirectory(atPath: directory.path) else { return false }
            let indexes = files.filter { $0.hasSuffix(".safetensors.index.json") }
            if indexes.isEmpty {
                return files.contains { $0.hasSuffix(".safetensors") && fileManager.fileExists(atPath: directory.appending(path: $0).path) }
            }
            return indexes.allSatisfy { index in
                guard let data = try? Data(contentsOf: directory.appending(path: index)),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let weightMap = json["weight_map"] as? [String: String], !weightMap.isEmpty
                else { return false }
                return Set(weightMap.values).allSatisfy { fileManager.fileExists(atPath: directory.appending(path: $0).path) }
            }
        }
    }
}
