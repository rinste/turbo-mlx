import Foundation

/// A repository as the hub's model API describes it: enough to tell, before downloading anything,
/// which family it is and whether its files make a checkpoint the engine reads.
nonisolated struct HubModel: Identifiable, Hashable, Sendable {
    let id: String
    let downloads: Int
    let likes: Int
    /// Its terms must be accepted on Hugging Face (and a token given) before it downloads.
    let gated: Bool
    /// The models it was made from, from its card's `base_model` (its `base_model:` tags).
    let baseModels: [String]
    /// Every file of the repository, by path.
    let files: [String]

    var author: String { id.split(separator: "/").first.map(String.init) ?? "" }
    var name: String { id.split(separator: "/").last.map(String.init) ?? id }

    init?(json: [String: Any]) {
        guard let id = json["id"] as? String else { return nil }
        self.id = id
        downloads = (json["downloads"] as? NSNumber)?.intValue ?? 0
        likes = (json["likes"] as? NSNumber)?.intValue ?? 0
        // false, or how access is granted: "auto" or "manual".
        gated = (json["gated"] as? String) != nil || (json["gated"] as? Bool) == true
        let tags = json["tags"] as? [String] ?? []
        var bases: [String] = []
        for tag in tags where tag.hasPrefix("base_model:") {
            // "base_model:quantized:org/name" and the like say how, then which.
            let base = tag.split(separator: ":").last.map(String.init) ?? ""
            if !base.isEmpty, !bases.contains(base) { bases.append(base) }
        }
        baseModels = bases
        files = (json["siblings"] as? [[String: Any]] ?? []).compactMap { $0["rfilename"] as? String }
    }
}

/// Finds checkpoints on Hugging Face: the repositories tagged `mflux`, the format the engine's
/// image families read.
nonisolated enum HubSearch {
    nonisolated enum SearchError: LocalizedError {
        case http(Int)
        case malformed

        var errorDescription: String? {
            switch self {
            case .http(let status): "Hugging Face answered \(status) to the search."
            case .malformed: "Hugging Face returned an unexpected answer."
            }
        }
    }

    /// The most downloaded `mflux` repositories whose name contains `query` (all of them when it
    /// is empty), with their tags and files.
    static func search(_ query: String) async throws -> [HubModel] {
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [
            URLQueryItem(name: "filter", value: "mflux"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "limit", value: "60"),
            URLQueryItem(name: "full", value: "true"),
        ]
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { components.queryItems?.append(URLQueryItem(name: "search", value: trimmed)) }
        guard let entries = try await json(components.url!) as? [[String: Any]] else { throw SearchError.malformed }
        return entries.compactMap(HubModel.init(json:))
    }

    /// One repository by its id, or nil when Hugging Face does not have it (or not for this token).
    static func model(_ repo: String) async throws -> HubModel? {
        let escaped = repo.split(separator: "/").map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
        let url = URL(string: "https://huggingface.co/api/models/" + escaped.joined(separator: "/"))!
        do {
            return (try await json(url) as? [String: Any]).flatMap(HubModel.init(json:))
        } catch SearchError.http(let status) where status == 401 || status == 404 {
            return nil
        }
    }

    /// The answer to a GET, sent with the token so the user's private repositories show too; a
    /// token the hub refuses (expired, revoked) is dropped rather than failing a public search.
    private static func json(_ url: URL) async throws -> Any {
        let token = HuggingFaceToken.current
        var (data, response) = try await URLSession.shared.data(for: HubDownloader.request(url, token: token))
        if token != nil, (response as? HTTPURLResponse)?.statusCode == 401 {
            (data, response) = try await URLSession.shared.data(for: HubDownloader.request(url, token: nil))
        }
        if let status = (response as? HTTPURLResponse)?.statusCode, !(200..<300).contains(status) {
            throw SearchError.http(status)
        }
        return try JSONSerialization.jsonObject(with: data)
    }
}
