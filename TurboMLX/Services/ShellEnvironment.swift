import Foundation

/// Apps launched from the Dock get a bare environment. Child processes should instead see what
/// the user's Terminal sees (PATH with uv/python, HF_HOME, HF_TOKEN…), so it is read once from an
/// interactive login shell.
nonisolated enum ShellEnvironment {
    /// Variables that would make our Python pick up someone else's packages.
    private static let pythonOverrides = [
        "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
        "CONDA_SHLVL", "__PYVENV_LAUNCHER__",
    ]

    @concurrent
    nonisolated static func resolve() async -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        if let shell = await loginShellEnvironment() {
            environment.merge(shell) { _, fromShell in fromShell }
        }
        for key in pythonOverrides { environment.removeValue(forKey: key) }

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        var path = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for extra in ["/opt/homebrew/bin", "/usr/local/bin", "\(home)/.local/bin", "\(home)/.cargo/bin",
                      "/usr/bin", "/bin", "/usr/sbin", "/sbin"] where !path.contains(extra) {
            path.append(extra)
        }
        environment["PATH"] = path.joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        return environment
    }

    @concurrent
    nonisolated private static func loginShellEnvironment() async -> [String: String]? {
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"
        let marker = "__TURBO_MLX_ENV__"
        // Output goes to a file rather than a pipe: a background job started by .zshrc could
        // otherwise hold the pipe open forever.
        let outputURL = FileManager.default.temporaryDirectory.appending(path: "turbo-mlx-env-\(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: outputURL.path, contents: nil),
              let output = try? FileHandle(forWritingTo: outputURL)
        else { return nil }
        defer {
            try? output.close()
            try? FileManager.default.removeItem(at: outputURL)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: shell)
        process.arguments = ["-ilc", "printf '\(marker)'; /usr/bin/env -0; printf '\(marker)'"]
        process.standardInput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.standardOutput = output

        let finished = AsyncStream<Void>.makeStream()
        process.terminationHandler = { _ in finished.continuation.finish() }
        do { try process.run() } catch { return nil }

        // A shell stuck on a prompt in .zshrc must not block the app.
        let pid = process.processIdentifier
        let watchdog = Task {
            try await Task.sleep(for: .seconds(8))
            kill(pid, SIGKILL)
        }
        for await _ in finished.stream {}
        watchdog.cancel()

        guard let data = try? Data(contentsOf: outputURL) else { return nil }
        let text = String(decoding: data, as: UTF8.self)
        let parts = text.components(separatedBy: marker)
        guard parts.count >= 3 else { return nil }
        var environment: [String: String] = [:]
        for entry in parts[1].split(separator: "\0") {
            guard let equals = entry.firstIndex(of: "=") else { continue }
            environment[String(entry[..<equals])] = String(entry[entry.index(after: equals)...])
        }
        // Shell bookkeeping that should not leak into child processes.
        for key in ["_", "SHLVL", "PWD", "OLDPWD"] { environment.removeValue(forKey: key) }
        return environment.isEmpty ? nil : environment
    }
}
