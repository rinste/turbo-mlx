import Foundation
import Security

/// The Hugging Face access token, for gated or private repositories (the catalog's models need
/// none): `HF_TOKEN` when the app was started with one, else the one saved in Settings → Models,
/// kept in the keychain, else, in the GitHub build, the one the huggingface CLI saved in
/// `~/.cache/huggingface/token`, which the sealed build's sandbox cannot read.
nonisolated enum HuggingFaceToken {
    private static let service = "io.github.rinste.TurboMLX.huggingface"
    private static let account = "access-token"

    static var current: String? {
        let environment = ProcessInfo.processInfo.environment
        if let token = environment["HF_TOKEN"] ?? environment["HUGGING_FACE_HUB_TOKEN"], !token.isEmpty { return token }
        if let saved { return saved }
        #if SEALED
        return nil
        #else
        let file = ModelFolder.realHome.appending(path: ".cache/huggingface/token")
        let stored = (try? String(contentsOf: file, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return stored.isEmpty ? nil : stored
        #endif
    }

    /// The token saved in Settings, if any.
    static var saved: String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess, let data = result as? Data,
              let token = String(data: data, encoding: .utf8), !token.isEmpty
        else { return nil }
        return token
    }

    /// Saves `token` in the keychain in place of the previous one.
    static func save(_ token: String) throws {
        remove()
        var item = baseQuery
        item[kSecValueData as String] = Data(token.utf8)
        item[kSecAttrLabel as String] = "Turbo MLX: Hugging Face token"
        let status = SecItemAdd(item as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: SecCopyErrorMessageString(status, nil) as String? ?? "Keychain error \(status)"])
        }
    }

    static func remove() {
        SecItemDelete(baseQuery as CFDictionary)
    }

    private static var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
    }
}
