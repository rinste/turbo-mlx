import Foundation

/// Downloads a Hugging Face repository into the hub cache, in the layout `huggingface_hub` uses:
/// `models--org--name/blobs/<etag>`, `snapshots/<commit>/<path>` linking into the blobs, and
/// `refs/main` for a branch. The files it fetches are therefore the ones mflux, `ModelLocator` and
/// other tools already see, and files they fetched are reused here. Interrupted downloads resume
/// from the bytes on disk, and a transfer cut off by the network (a dropped connection, a Mac waking
/// up) is tried again on its own for a couple of minutes before the download fails.
nonisolated final class HubDownloader: Sendable {
    nonisolated struct RemoteFile: Sendable, Hashable {
        let path: String
        let size: Int64
        /// The blob's name in the cache: the LFS sha256 for large files, the git blob id otherwise,
        /// which is what the hub returns as the file's ETag.
        let etag: String
    }

    nonisolated struct Listing: Sendable {
        let commit: String
        let files: [RemoteFile]
        var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }
    }

    nonisolated enum DownloadError: LocalizedError {
        case http(Int, String)
        case malformedListing
        case truncated(String, expected: Int64, got: Int64)

        var errorDescription: String? {
            switch self {
            case .http(401, let repo):
                "\(repo) needs a Hugging Face token: the repository is gated or private. Sign in with the Hugging Face CLI (hf auth login): Turbo MLX uses the token it saves."
            // A gated repository answers 403 to a token whose account has not accepted its terms.
            case .http(403, let repo):
                "\(repo) is gated: open huggingface.co/\(repo) signed in to the account of your Hugging Face token and accept its terms, then download again. Without a token, sign in with the Hugging Face CLI (hf auth login) first."
            case .http(404, let repo):
                "\(repo) was not found on Hugging Face."
            case .http(let status, let repo):
                "Hugging Face answered \(status) for \(repo)."
            case .malformedListing:
                "Hugging Face returned an unexpected file listing."
            case .truncated(let path, let expected, let got):
                "\(path) is incomplete: \(got) of \(expected) bytes. The download resumes next time."
            }
        }
    }

    let repo: String
    /// A branch, or the commit a catalog model was checked with.
    let revision: String
    let hubCache: URL
    private let token: String?
    private let session: URLSession
    /// Files fetched at the same time: the hub's CDN gives several connections more than one.
    private let parallelism = 4
    /// The waits before each new attempt after a transient failure: about two minutes in all.
    private static let retryDelays: [Duration] = [.seconds(2), .seconds(5), .seconds(10), .seconds(20), .seconds(30), .seconds(60)]

    init(repo: String, hubCache: URL, revision: String = "main") {
        self.repo = repo
        self.revision = revision
        self.hubCache = hubCache
        token = HuggingFaceToken.current
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 60 * 60 * 24
        configuration.httpMaximumConnectionsPerHost = 6
        session = URLSession(configuration: configuration)
    }

    // MARK: Layout

    var repoFolder: URL {
        hubCache.appending(path: "models--" + repo.replacingOccurrences(of: "/", with: "--"), directoryHint: .isDirectory)
    }

    private var blobs: URL { repoFolder.appending(path: "blobs", directoryHint: .isDirectory) }

    func snapshotFolder(commit: String) -> URL {
        repoFolder.appending(path: "snapshots", directoryHint: .isDirectory).appending(path: commit, directoryHint: .isDirectory)
    }

    /// True when the blob is on disk with the expected size.
    func isCached(_ file: RemoteFile) -> Bool {
        let values = try? blobs.appending(path: file.etag).resourceValues(forKeys: [.fileSizeKey])
        return values?.fileSize.map { Int64($0) == file.size } ?? false
    }

    /// The bytes still to fetch: the files not cached, less what an earlier attempt left of them.
    func bytesMissing(from listing: Listing) -> Int64 {
        listing.files.filter { !isCached($0) }.reduce(0) { total, file in
            let partial = blobs.appending(path: file.etag + ".incomplete")
            let kept = Int64((try? partial.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            // A partial file as long as the whole one is fetched again (see `fetchOnce`).
            return total + file.size - (kept < file.size ? kept : 0)
        }
    }

    // MARK: Listing

    /// The repository's files matching `patterns` (fnmatch style, as huggingface_hub reads them:
    /// `*` also matches `/`), and the commit they belong to.
    func list(matching patterns: [String]) async throws -> Listing {
        try await retrying { try await self.listOnce(matching: patterns) }
    }

    private func listOnce(matching patterns: [String]) async throws -> Listing {
        let info = try await json(from: "https://huggingface.co/api/models/\(repo)/revision/\(revision)")
        guard let commit = info["sha"] as? String else { throw DownloadError.malformedListing }
        let regexes = patterns.map(Self.regex(fnmatch:))
        var files: [RemoteFile] = []
        var page: String? = "https://huggingface.co/api/models/\(repo)/tree/\(commit)?recursive=true"
        while let url = page {
            let (entries, next) = try await treePage(url)
            for entry in entries {
                guard entry["type"] as? String == "file", let path = entry["path"] as? String else { continue }
                let range = NSRange(path.startIndex..., in: path)
                guard regexes.isEmpty || regexes.contains(where: { $0.firstMatch(in: path, range: range) != nil }) else { continue }
                let lfs = entry["lfs"] as? [String: Any]
                let size = ((lfs?["size"] as? NSNumber) ?? (entry["size"] as? NSNumber))?.int64Value ?? 0
                guard let etag = (lfs?["oid"] as? String) ?? (entry["oid"] as? String) else { continue }
                files.append(RemoteFile(path: path, size: size, etag: etag))
            }
            page = next
        }
        return Listing(commit: commit, files: files)
    }

    private func treePage(_ urlString: String) async throws -> ([[String: Any]], String?) {
        let (data, response) = try await session.data(for: request(urlString))
        try check(response)
        guard let entries = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw DownloadError.malformedListing
        }
        // Long listings continue at the URL of the Link header's rel="next".
        var next: String?
        if let link = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Link"),
           let range = link.range(of: #"<[^>]+>;\s*rel="next""#, options: .regularExpression) {
            let match = link[range]
            if let close = match.firstIndex(of: ">") {
                next = String(match[match.index(after: match.startIndex)..<close])
            }
        }
        return (entries, next)
    }

    private func json(from urlString: String) async throws -> [String: Any] {
        let (data, response) = try await session.data(for: request(urlString))
        try check(response)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DownloadError.malformedListing
        }
        return object
    }

    /// Python's fnmatch as a regular expression: `*` (and `**`) match anything, `?` one character.
    static func regex(fnmatch pattern: String) -> NSRegularExpression {
        var expression = "^"
        for character in pattern {
            switch character {
            case "*": expression += ".*"
            case "?": expression += "."
            default: expression += NSRegularExpression.escapedPattern(for: String(character))
            }
        }
        expression += "$"
        // Built from a fixed alphabet, so it is always valid.
        return try! NSRegularExpression(pattern: expression)
    }

    // MARK: Download

    /// Fetches every file of `listing` that is not cached yet and links the snapshot. `progress`
    /// receives the bytes on disk so far, cached ones included. Returns the snapshot folder.
    func download(_ listing: Listing, progress: @escaping @Sendable (Int64) -> Void) async throws -> URL {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: blobs, withIntermediateDirectories: true)
        let snapshot = snapshotFolder(commit: listing.commit)
        try fileManager.createDirectory(at: snapshot, withIntermediateDirectories: true)

        let counter = ByteCounter(initial: listing.files.filter(isCached).reduce(0) { $0 + $1.size }, report: progress)
        counter.report()
        let pending = listing.files.filter { !isCached($0) }

        try await withThrowingTaskGroup(of: Void.self) { group in
            var next = 0
            var running = 0
            while running < parallelism, next < pending.count {
                let file = pending[next]
                next += 1
                running += 1
                group.addTask { try await self.fetch(file, commit: listing.commit, counter: counter) }
            }
            while running > 0 {
                try await group.next()
                running -= 1
                if next < pending.count {
                    let file = pending[next]
                    next += 1
                    running += 1
                    group.addTask { try await self.fetch(file, commit: listing.commit, counter: counter) }
                }
            }
        }

        for file in listing.files { try link(file, in: snapshot) }
        // A branch points at its commit, as huggingface_hub records it; a commit needs no ref.
        if revision != listing.commit {
            let refs = repoFolder.appending(path: "refs", directoryHint: .isDirectory)
            try fileManager.createDirectory(at: refs, withIntermediateDirectories: true)
            try listing.commit.write(to: refs.appending(path: revision), atomically: true, encoding: .utf8)
        }
        return snapshot
    }

    /// Fetches one file of `commit` into `blobs/<etag>`, trying again after a transient failure.
    private func fetch(_ file: RemoteFile, commit: String, counter: ByteCounter) async throws {
        try await retrying { try await self.fetchOnce(file, commit: commit, counter: counter) }
    }

    /// One attempt, resuming the `.incomplete` file an earlier one left.
    private func fetchOnce(_ file: RemoteFile, commit: String, counter: ByteCounter) async throws {
        let fileManager = FileManager.default
        let destination = blobs.appending(path: file.etag)
        let partial = blobs.appending(path: file.etag + ".incomplete")
        var offset = Int64((try? partial.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        if offset >= file.size {
            // Everything arrived last time but the file was not renamed: fetch it again cleanly.
            offset = 0
            try? fileManager.removeItem(at: partial)
        }
        if !fileManager.fileExists(atPath: partial.path) {
            fileManager.createFile(atPath: partial.path, contents: nil)
        }

        let escaped = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
        var request = request("https://huggingface.co/\(repo)/resolve/\(commit)/\(escaped)")
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

        let handle = try FileHandle(forWritingTo: partial)
        defer { try? handle.close() }
        try handle.seekToEnd()
        counter.add(offset)
        let download = ChunkedDownload(handle: handle, resumingFrom: offset) { counter.add($0) }
        do {
            let response = try await download.run(request, in: session)
            if let response, !(200..<300).contains(response.statusCode) {
                throw DownloadError.http(response.statusCode, repo)
            }
            try handle.close()
            guard download.written == file.size else {
                throw DownloadError.truncated(file.path, expected: file.size, got: download.written)
            }
        } catch {
            // The bytes counted here are on disk, and the next attempt counts them from the file.
            counter.add(-download.written)
            throw error
        }
        if fileManager.fileExists(atPath: destination.path) { try fileManager.removeItem(at: destination) }
        try fileManager.moveItem(at: partial, to: destination)
    }

    /// Runs `attempt` again after a failure of the network or the server, waiting longer each
    /// time; any other error, and cancellation, ends it at once.
    private func retrying<T>(_ attempt: () async throws -> T) async throws -> T {
        for delay in Self.retryDelays {
            do {
                return try await attempt()
            } catch let error where Self.isTransient(error) {
                try await Task.sleep(for: delay)
            }
        }
        return try await attempt()
    }

    /// What another attempt can fix: a connection that dropped or timed out, no network for a
    /// moment, a server that is busy or restarting. Not a missing file, a refused token or a full disk.
    static func isTransient(_ error: any Error) -> Bool {
        switch error {
        case let error as URLError:
            [.timedOut, .networkConnectionLost, .notConnectedToInternet, .cannotConnectToHost, .cannotFindHost,
             .dnsLookupFailed, .secureConnectionFailed, .dataNotAllowed, .internationalRoamingOff, .callIsActive]
                .contains(error.code)
        case DownloadError.http(let status, _):
            status == 408 || status == 429 || (500...599).contains(status)
        case DownloadError.truncated:
            true
        default:
            false
        }
    }

    /// `snapshots/<commit>/<path>` → `../../blobs/<etag>`, relative like huggingface_hub's links.
    private func link(_ file: RemoteFile, in snapshot: URL) throws {
        let fileManager = FileManager.default
        let target = snapshot.appending(path: file.path)
        try fileManager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        if (try? target.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true
            || fileManager.fileExists(atPath: target.path) {
            try fileManager.removeItem(at: target)
        }
        let depth = file.path.split(separator: "/").count
        let up = Array(repeating: "..", count: depth + 1).joined(separator: "/")
        try fileManager.createSymbolicLink(atPath: target.path, withDestinationPath: "\(up)/blobs/\(file.etag)")
    }

    // MARK: Requests

    private func request(_ urlString: String) -> URLRequest {
        var request = URLRequest(url: URL(string: urlString)!)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.setValue("turbo-mlx", forHTTPHeaderField: "User-Agent")
        return request
    }

    private func check(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else { throw DownloadError.http(http.statusCode, repo) }
    }
}

/// One HTTP body written to a file as URLSession delivers it, in chunks. (Iterating `AsyncBytes`
/// one byte at a time costs a core on multi-gigabyte files.)
nonisolated private final class ChunkedDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let handle: FileHandle
    private let resumeOffset: Int64
    private let onChunk: @Sendable (Int64) -> Void
    private var continuation: CheckedContinuation<HTTPURLResponse?, any Error>?
    private var response: HTTPURLResponse?
    private var failure: (any Error)?
    /// Bytes in the file once the transfer ends: the resumed ones plus what arrived.
    private(set) var written: Int64

    init(handle: FileHandle, resumingFrom offset: Int64, onChunk: @escaping @Sendable (Int64) -> Void) {
        self.handle = handle
        resumeOffset = offset
        written = offset
        self.onChunk = onChunk
    }

    /// Runs the request to completion and returns the server's answer (200, or 206 when resuming).
    func run(_ request: URLRequest, in session: URLSession) async throws -> HTTPURLResponse? {
        let task = session.dataTask(with: request)
        task.delegate = self
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let http = response as? HTTPURLResponse
        self.response = http
        guard let http, (200..<300).contains(http.statusCode) else {
            completionHandler(.cancel)
            return
        }
        if resumeOffset > 0, http.statusCode != 206 {
            // The server ignored the range and is sending the whole file: start the file over.
            do {
                try handle.truncate(atOffset: 0)
                onChunk(-resumeOffset)
                written = 0
            } catch {
                failure = error
                completionHandler(.cancel)
                return
            }
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle.write(contentsOf: data)
            written += Int64(data.count)
            onChunk(Int64(data.count))
        } catch {
            failure = error
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        guard let continuation else { return }
        self.continuation = nil
        if let failure {
            continuation.resume(throwing: failure)
        } else if let response, !(200..<300).contains(response.statusCode) {
            continuation.resume(returning: response)
        } else if let error {
            continuation.resume(throwing: error)
        } else {
            continuation.resume(returning: response)
        }
    }
}

/// Bytes on disk across parallel transfers, reported to the caller a few times a second.
nonisolated private final class ByteCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes: Int64
    private let deliver: @Sendable (Int64) -> Void
    private var lastReport = Date.distantPast

    init(initial: Int64, report: @escaping @Sendable (Int64) -> Void) {
        bytes = initial
        deliver = report
    }

    func add(_ count: Int64) {
        lock.lock()
        bytes += count
        let now = Date()
        let due = now.timeIntervalSince(lastReport) >= 0.3
        if due { lastReport = now }
        let value = bytes
        lock.unlock()
        if due { deliver(value) }
    }

    func report() {
        lock.lock()
        let value = bytes
        lastReport = Date()
        lock.unlock()
        deliver(value)
    }
}
