import Foundation
import Observation

/// A child process whose stdout and stderr are delivered line by line on the main actor.
final class LineProcess {
    private let process = Process()
    private let input = Pipe()

    init(executable: URL, arguments: [String], environment: [String: String]) {
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
    }

    var isRunning: Bool { process.isRunning }

    /// Starts the process. `onExit` runs once both output streams are drained, with the exit status.
    func start(
        onStdout: @escaping (String) -> Void,
        onStderr: @escaping (String) -> Void,
        onExit: @escaping (Int32) -> Void
    ) throws {
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardInput = input
        process.standardOutput = stdout
        process.standardError = stderr

        let (exitStatus, exitContinuation) = AsyncStream<Int32>.makeStream()
        process.terminationHandler = { process in
            exitContinuation.yield(process.terminationStatus)
            exitContinuation.finish()
        }
        try process.run()

        let stdoutLines = stdout.fileHandleForReading.lines()
        let stderrLines = stderr.fileHandleForReading.lines()
        let readStdout = Task { for await line in stdoutLines { onStdout(line) } }
        let readStderr = Task { for await line in stderrLines { onStderr(line) } }
        Task {
            await readStdout.value
            await readStderr.value
            var status: Int32 = -1
            for await value in exitStatus { status = value }
            onExit(status)
        }
    }

    func send(line: String) throws {
        try input.fileHandleForWriting.write(contentsOf: Data((line + "\n").utf8))
    }

    func terminate() {
        guard process.isRunning else { return }
        process.terminate()
    }

    /// SIGTERM first; SIGKILL if the process is still alive after `grace` seconds.
    func stop(grace: Duration = .seconds(3)) {
        guard process.isRunning else { return }
        let pid = process.processIdentifier
        process.terminate()
        Task.detached {
            try? await Task.sleep(for: grace)
            if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        }
    }
}

extension FileHandle {
    /// Lines read from the handle until EOF. A carriage return rewinds the line like a
    /// terminal would, so tqdm progress bars collapse into their final state.
    nonisolated func lines() -> AsyncStream<String> {
        AsyncStream { continuation in
            let splitter = LineSplitter()
            readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty {
                    handle.readabilityHandler = nil
                    splitter.flush().forEach { continuation.yield($0) }
                    continuation.finish()
                } else {
                    splitter.append(data).forEach { continuation.yield($0) }
                }
            }
        }
    }
}

/// Accumulates bytes and cuts them into lines. Only touched from one FileHandle's serial handler queue.
nonisolated private final class LineSplitter: @unchecked Sendable {
    private var buffer = Data()

    func append(_ data: Data) -> [String] {
        buffer.append(data)
        var lines: [String] = []
        while let newline = buffer.firstIndex(of: UInt8(ascii: "\n")) {
            lines.append(Self.decode(buffer[buffer.startIndex..<newline]))
            buffer.removeSubrange(buffer.startIndex...newline)
        }
        return lines
    }

    func flush() -> [String] {
        defer { buffer.removeAll() }
        return buffer.isEmpty ? [] : [Self.decode(buffer)]
    }

    private static func decode(_ bytes: Data) -> String {
        let line = String(decoding: bytes, as: UTF8.self)
        guard let lastReturn = line.dropLast(line.hasSuffix("\r") ? 1 : 0).lastIndex(of: "\r") else {
            return line.trimmingCharacters(in: .newlines)
        }
        return String(line[line.index(after: lastReturn)...]).trimmingCharacters(in: .newlines)
    }
}

/// The tail of every backend process's stderr, for the log window and error reports.
@Observable
final class LogBuffer {
    struct Line: Identifiable {
        let id: Int
        let date: Date
        let text: String
    }

    private(set) var lines: [Line] = []
    private var nextID = 0
    private let capacity = 4000

    func append(_ text: String) {
        guard !text.isEmpty else { return }
        lines.append(Line(id: nextID, date: Date(), text: text))
        nextID += 1
        if lines.count > capacity { lines.removeFirst(lines.count - capacity) }
    }

    func clear() { lines.removeAll() }

    func tail(_ count: Int) -> String {
        lines.suffix(count).map(\.text).joined(separator: "\n")
    }

    var text: String { lines.map(\.text).joined(separator: "\n") }
}
