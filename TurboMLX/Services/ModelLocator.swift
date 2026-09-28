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
    /// A family with a companion checkpoint (LTX-2's text encoder) needs that one too.
    func installedLocation(of model: ModelDescriptor) -> URL? {
        if model.family.companion != nil, companionLocation(of: model) == nil { return nil }
        switch model.source {
        case .local(let path):
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            return Self.isComplete(url, family: model.family) ? url : nil
        case .huggingFace(let repo):
            return newestSnapshot(of: repo) { Self.isComplete($0, family: model.family) }
        }
    }

    /// The companion checkpoint's folder (LTX-2: Gemma 3 12B), when it is complete.
    func companionLocation(of model: ModelDescriptor) -> URL? {
        guard let companion = model.family.companion else { return nil }
        return newestSnapshot(of: companion.repo) { Self.hasFiles(companion.requiredFiles, in: $0) }
    }

    private func newestSnapshot(of repo: String, where isComplete: (URL) -> Bool) -> URL? {
        let snapshots = repoFolder(repo).appending(path: "snapshots")
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        guard let entries = try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: keys)
        else { return nil }
        let newestFirst = entries.sorted {
            let lhs = (try? $0.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
            let rhs = (try? $1.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
            return lhs > rhs
        }
        return newestFirst.first(where: isComplete)
    }

    /// Each name present (a `*` matches any run of characters), as a file and not a dangling link.
    static func hasFiles(_ names: [String], in folder: URL) -> Bool {
        let fileManager = FileManager.default
        let files = (try? fileManager.contentsOfDirectory(atPath: folder.path)) ?? []
        return names.allSatisfy { name in
            let regex = HubDownloader.regex(fnmatch: name)
            return files.contains { file in
                regex.firstMatch(in: file, range: NSRange(file.startIndex..., in: file)) != nil
                    && fileManager.fileExists(atPath: folder.appending(path: file).path)
            }
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
        guard hasFiles(family.requiredFiles, in: folder) else { return false }
        if let tokenizer = family.tokenizerFile, !fileManager.fileExists(atPath: folder.appending(path: tokenizer).path) {
            return false
        }
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
