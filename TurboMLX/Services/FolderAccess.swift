import Foundation

/// Folders the user picked for local models. The sandbox lets the app into a folder outside the
/// Hugging Face cache only through the security-scoped bookmark kept here when it was chosen; the
/// access is opened at launch, before the engine starts, so the engine (a child process in the
/// app's sandbox) inherits it and can read the models too.
enum FolderAccess {
    private static let key = "folderBookmarks"

    /// Keeps a bookmark to a folder just chosen in an open panel and opens its access.
    static func remember(_ url: URL) {
        guard let bookmark = try? url.bookmarkData(options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess])
        else { return }
        var bookmarks = stored
        bookmarks[url.path(percentEncoded: false)] = bookmark
        UserDefaults.standard.set(bookmarks, forKey: key)
        _ = url.startAccessingSecurityScopedResource()
    }

    static func forget(_ path: String) {
        var bookmarks = stored
        bookmarks[path] = nil
        UserDefaults.standard.set(bookmarks, forKey: key)
    }

    /// Opens every remembered folder for the life of the app, renewing stale bookmarks.
    static func restore() {
        var bookmarks = stored
        for (path, bookmark) in bookmarks {
            var isStale = false
            guard let url = try? URL(resolvingBookmarkData: bookmark, options: .withSecurityScope, bookmarkDataIsStale: &isStale),
                  url.startAccessingSecurityScopedResource()
            else { continue }
            if isStale, let renewed = try? url.bookmarkData(options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess]) {
                bookmarks[path] = renewed
            }
        }
        UserDefaults.standard.set(bookmarks, forKey: key)
    }

    private static var stored: [String: Data] {
        UserDefaults.standard.dictionary(forKey: key) as? [String: Data] ?? [:]
    }
}
