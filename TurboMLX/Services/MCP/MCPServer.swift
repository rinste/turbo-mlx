#if !SEALED
import Foundation
import Observation

/// Lets agents on this Mac use the app: an MCP server (Model Context Protocol, over its
/// Streamable HTTP transport) that lists the models, queues generations next to the window's
/// own and says how they are going. Off until turned on in Settings; it listens on 127.0.0.1
/// only, and answers nothing without its token.
@Observable
final class MCPServer {
    enum Status: Equatable {
        case off
        case starting
        case listening
        case failed(String)
    }

    static let defaultPort = 8733
    static let path = "/mcp"
    /// The protocol versions answered as asked; any other gets the first.
    private static let versions = ["2025-06-18", "2025-11-25", "2025-03-26"]

    private enum Keys {
        static let enabled = "mcpEnabled"
        static let port = "mcpPort"
        static let token = "mcpToken"
    }

    private(set) var status = Status.off
    private(set) var token: String

    var isEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnabled, forKey: Keys.enabled)
            restart()
        }
    }

    var port: Int {
        didSet {
            guard port != oldValue else { return }
            UserDefaults.standard.set(port, forKey: Keys.port)
            restart()
        }
    }

    var address: String { "http://127.0.0.1:\(port)\(Self.path)" }

    @ObservationIgnored weak var app: AppModel?
    @ObservationIgnored private var listener: HTTPListener?
    @ObservationIgnored private var starts = 0
    /// The jobs agents queued, to say how each ended once it has left the queue.
    @ObservationIgnored var jobs: [UUID: GenerationJob] = [:]

    init() {
        let defaults = UserDefaults.standard
        isEnabled = defaults.bool(forKey: Keys.enabled)
        port = defaults.object(forKey: Keys.port) as? Int ?? Self.defaultPort
        if let saved = defaults.string(forKey: Keys.token) {
            token = saved
        } else {
            token = Self.newToken()
            defaults.set(token, forKey: Keys.token)
        }
    }

    /// Called once the app has started: the server needs it to answer anything.
    func attach(_ app: AppModel) {
        self.app = app
        restart()
    }

    /// Another token: the agents set up with the old one are turned away until they get it.
    func renewToken() {
        token = Self.newToken()
        UserDefaults.standard.set(token, forKey: Keys.token)
    }

    func stop() {
        starts += 1
        listener?.stop()
        listener = nil
        status = .off
    }

    private static func newToken() -> String {
        (0..<32).map { _ in String(format: "%02x", UInt8.random(in: .min ... .max)) }.joined()
    }

    private func restart() {
        stop()
        guard isEnabled, app != nil else { return }
        guard let number = UInt16(exactly: port), number >= 1024 else {
            status = .failed("Choose a port between 1024 and 65535.")
            return
        }
        status = .starting
        // A listener stopped since must not report over the one that replaced it.
        starts += 1
        let start = starts
        do {
            listener = try HTTPListener(
                port: number,
                handler: { [weak self] request in
                    await self?.respond(to: request) ?? HTTPResponse(status: 503)
                },
                onState: { [weak self] result in
                    guard let self, starts == start else { return }
                    switch result {
                    case .success: status = .listening
                    case .failure(let error): status = .failed(error.localizedDescription)
                    }
                }
            )
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    // MARK: HTTP

    private func respond(to request: HTTPRequest) async -> HTTPResponse {
        guard request.path == Self.path else { return HTTPResponse(status: 404) }
        // A web page must not reach the server through the browser: only pages of this Mac may.
        if let origin = request.headers["origin"] {
            let host = URLComponents(string: origin)?.host
            guard ["localhost", "127.0.0.1", "::1", "[::1]"].contains(host) else { return HTTPResponse(status: 403) }
        }
        guard request.headers["authorization"] == "Bearer \(token)" else {
            return HTTPResponse(status: 401, headers: ["WWW-Authenticate": "Bearer"])
        }
        // The server has nothing to say unasked, so there is no stream to open (GET) and no
        // session to end (DELETE).
        guard request.method == "POST" else { return HTTPResponse(status: 405, headers: ["Allow": "POST"]) }

        guard let message = try? JSONSerialization.jsonObject(with: request.body) else {
            return .json(400, Self.data(Self.failure(id: NSNull(), code: -32700, message: "Parse error")))
        }
        if let batch = message as? [[String: Any]] {
            var answers: [[String: Any]] = []
            for item in batch {
                if let answer = await answer(item) { answers.append(answer) }
            }
            return answers.isEmpty ? HTTPResponse(status: 202) : .json(200, Self.data(answers))
        }
        guard let single = message as? [String: Any] else {
            return .json(400, Self.data(Self.failure(id: NSNull(), code: -32600, message: "Invalid request")))
        }
        guard let answer = await answer(single) else { return HTTPResponse(status: 202) }
        return .json(200, Self.data(answer))
    }

    // MARK: JSON-RPC

    /// The answer to a request; nil for a notification or a client's own answer, which get none.
    private func answer(_ message: [String: Any]) async -> [String: Any]? {
        guard let method = message["method"] as? String, let id = message["id"], !(id is NSNull) else { return nil }
        let params = message["params"] as? [String: Any] ?? [:]
        switch method {
        case "initialize":
            let asked = params["protocolVersion"] as? String
            let version = asked.flatMap { Self.versions.contains($0) ? $0 : nil } ?? Self.versions[0]
            return Self.success(id: id, [
                "protocolVersion": version,
                "capabilities": ["tools": [String: Any]()],
                "serverInfo": [
                    "name": "turbo-mlx",
                    "title": "Turbo MLX",
                    "version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0",
                ],
                "instructions": "Generates images, video, songs and sounds on this Mac with the Turbo MLX app. "
                    + "list_models first; generate returns a job at once; get_job (with wait_seconds) says when its file is ready.",
            ])
        case "ping":
            return Self.success(id: id, [:])
        case "tools/list":
            return Self.success(id: id, ["tools": Self.tools])
        case "tools/call":
            guard let name = params["name"] as? String, Self.tools.contains(where: { $0["name"] as? String == name }) else {
                return Self.failure(id: id, code: -32602, message: "Unknown tool")
            }
            do {
                let result = try await call(name, params["arguments"] as? [String: Any] ?? [:])
                return Self.success(id: id, [
                    "content": [["type": "text", "text": String(decoding: Self.data(result, pretty: true), as: UTF8.self)]],
                    "structuredContent": result,
                    "isError": false,
                ])
            } catch {
                // What went wrong with the tool is for the agent to read and act on: a result, not a protocol error.
                return Self.success(id: id, [
                    "content": [["type": "text", "text": error.localizedDescription]],
                    "isError": true,
                ])
            }
        default:
            return Self.failure(id: id, code: -32601, message: "Method not found")
        }
    }

    private static func success(id: Any, _ result: [String: Any]) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "result": result]
    }

    private static func failure(id: Any, code: Int, message: String) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]]
    }

    private static func data(_ object: Any, pretty: Bool = false) -> Data {
        (try? JSONSerialization.data(withJSONObject: object, options: pretty ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes] : [.withoutEscapingSlashes])) ?? Data("{}".utf8)
    }
}
#endif
