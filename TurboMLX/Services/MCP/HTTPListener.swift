#if !SEALED
import Foundation
import Network

/// A request the MCP server got: the header names in lower case.
nonisolated struct HTTPRequest: Sendable {
    var method: String
    var path: String
    var headers: [String: String]
    var body: Data
}

nonisolated struct HTTPResponse: Sendable {
    var status: Int
    var headers: [String: String] = [:]
    var body = Data()

    static func json(_ status: Int = 200, _ body: Data) -> HTTPResponse {
        HTTPResponse(status: status, headers: ["Content-Type": "application/json"], body: body)
    }

    fileprivate var encoded: Data {
        let reasons = [200: "OK", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
                       404: "Not Found", 405: "Method Not Allowed", 413: "Content Too Large", 503: "Service Unavailable"]
        var head = "HTTP/1.1 \(status) \(reasons[status] ?? "Status")\r\n"
        for (name, value) in headers { head += "\(name): \(value)\r\n" }
        head += "Content-Length: \(body.count)\r\n\r\n"
        return Data(head.utf8) + body
    }
}

/// The small HTTP/1.1 server under the MCP one: this Mac only (it listens on 127.0.0.1), a
/// request at a time on each connection, bodies with a Content-Length. Enough for the JSON an
/// agent posts, and nothing the app has to depend on.
nonisolated final class HTTPListener: @unchecked Sendable {
    typealias Handler = @Sendable (HTTPRequest) async -> HTTPResponse

    /// The longest request read: an agent's JSON is a few hundred bytes.
    private static let limit = 1 << 20

    private let listener: NWListener
    private let queue = DispatchQueue(label: "io.github.rinste.TurboMLX.mcp")
    private let handler: Handler

    /// `onState` is told, on the main actor, that the port is open or why it is not.
    init(port: UInt16, handler: @escaping Handler, onState: @escaping @MainActor (Result<Void, Error>) -> Void) throws {
        guard let port = NWEndpoint.Port(rawValue: port) else { throw POSIXError(.EINVAL) }
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: port)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
        self.handler = handler
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: Task { @MainActor in onState(.success(())) }
            case .failed(let error), .waiting(let error): Task { @MainActor in onState(.failure(error)) }
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return connection.cancel() }
            connection.start(queue: queue)
            serve(connection, buffer: Data())
        }
        listener.start(queue: queue)
    }

    func stop() {
        listener.cancel()
    }

    /// Answers the request `buffer` holds, or reads on until it holds one.
    private func serve(_ connection: NWConnection, buffer: Data) {
        switch Self.parse(buffer) {
        case .request(let request, let rest):
            Task { [self] in
                let response = await handler(request)
                connection.send(content: response.encoded, completion: .contentProcessed { error in
                    guard error == nil else { return connection.cancel() }
                    self.queue.async { self.serve(connection, buffer: rest) }
                })
            }
        case .invalid(let status):
            connection.send(content: HTTPResponse(status: status, headers: ["Connection": "close"]).encoded,
                            completion: .contentProcessed { _ in connection.cancel() })
        case .incomplete:
            connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, isComplete, error in
                guard let self, let data, !data.isEmpty, error == nil else { return connection.cancel() }
                if isComplete, case .incomplete = Self.parse(buffer + data) { return connection.cancel() }
                serve(connection, buffer: buffer + data)
            }
        }
    }

    private enum Parsed {
        case request(HTTPRequest, rest: Data)
        case incomplete
        case invalid(Int)
    }

    private static func parse(_ buffer: Data) -> Parsed {
        guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            return buffer.count > limit ? .invalid(413) : .incomplete
        }
        guard let head = String(data: buffer[buffer.startIndex..<end.lowerBound], encoding: .utf8) else { return .invalid(400) }
        var lines = head.components(separatedBy: "\r\n")
        let start = lines.removeFirst().split(separator: " ")
        guard start.count == 3 else { return .invalid(400) }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { return .invalid(400) }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        // A body sent in chunks has no length to wait for: none of the MCP clients sends its JSON that way.
        guard headers["transfer-encoding"] == nil else { return .invalid(400) }
        let length = headers["content-length"].flatMap { Int($0) } ?? 0
        guard length >= 0, length <= limit else { return .invalid(413) }
        let bodyStart = end.upperBound
        guard buffer.endIndex - bodyStart >= length else { return .incomplete }
        let bodyEnd = bodyStart + length
        let request = HTTPRequest(
            method: String(start[0]),
            path: String(start[1].split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)[0]),
            headers: headers,
            body: Data(buffer[bodyStart..<bodyEnd])
        )
        return .request(request, rest: Data(buffer[bodyEnd...]))
    }
}
#endif
