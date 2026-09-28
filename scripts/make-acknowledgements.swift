// Writes TurboMLX/Resources/Acknowledgements.txt, the licenses of the code in the app, which
// Settings → About shows: the projects the engine's ports follow (Engine/Licenses), the engine's
// Swift packages (Engine/Package.resolved, read from the checkouts of the engine's last build)
// and the libraries MLX carries inside mlx-swift. The Apache License is written out once, with
// each package's NOTICE. Run it after changing the engine's dependencies:
//
//   scripts/build-engine.sh && swift scripts/make-acknowledgements.swift

import Foundation

let checkouts = "build/engine/SourcePackages/checkouts"
let output = "TurboMLX/Resources/Acknowledgements.txt"

struct Component {
    let name: String
    let url: String
    let version: String?
    /// The folder holding its LICENSE (and NOTICE), or the license file itself.
    let licensePath: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("make-acknowledgements: \(message)\n".utf8))
    exit(1)
}

func read(_ path: String) -> String {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { fail("cannot read \(path)") }
    return text.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// The first file in `folder` whose name starts with one of `prefixes` (LICENSE, LICENSE.txt…).
func file(in folder: String, named prefixes: [String]) -> String? {
    let names = ((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).sorted()
    for prefix in prefixes {
        if let name = names.first(where: { $0.uppercased().hasPrefix(prefix) }) { return "\(folder)/\(name)" }
    }
    return nil
}

func isApache(_ text: String) -> Bool {
    text.contains("Apache License") && text.contains("Version 2.0")
}

// The projects the ports follow: Engine/Licenses/<name>.txt.
let ported: [(name: String, url: String, use: String)] = [
    ("mflux", "https://github.com/mflux-community/mflux", "the image families, module for module"),
    ("ltx-2-mlx", "https://github.com/dgrauet/ltx-2-mlx", "LTX-2.3, video and sound"),
    ("mlx-lm", "https://github.com/ml-explore/mlx-lm", "the mixture-of-experts layers"),
    ("mlx-swift-lm", "https://github.com/ml-explore/mlx-swift-lm", "the mixture-of-experts layers"),
]

// The engine's packages, as resolved.
guard let resolvedData = FileManager.default.contents(atPath: "Engine/Package.resolved"),
      let resolved = try? JSONSerialization.jsonObject(with: resolvedData) as? [String: Any],
      let pins = resolved["pins"] as? [[String: Any]]
else { fail("cannot read Engine/Package.resolved") }
guard FileManager.default.fileExists(atPath: checkouts) else { fail("no \(checkouts): run scripts/build-engine.sh first") }

var packages: [Component] = pins.map { pin in
    let identity = pin["identity"] as? String ?? "?"
    let location = (pin["location"] as? String ?? "").replacingOccurrences(of: ".git", with: "")
    let version = (pin["state"] as? [String: Any])?["version"] as? String
    return Component(name: location.split(separator: "/").last.map(String.init) ?? identity, url: location, version: version,
                     licensePath: "\(checkouts)/\(identity)")
}
packages.sort { $0.name.lowercased() < $1.name.lowercased() }

// Inside mlx-swift: the MLX core and the libraries it compiles in.
let cmlx = "\(checkouts)/mlx-swift/Source/Cmlx"
let bundled: [Component] = [
    Component(name: "MLX", url: "https://github.com/ml-explore/mlx", version: nil, licensePath: "\(cmlx)/mlx"),
    Component(name: "MLX C", url: "https://github.com/ml-explore/mlx-c", version: nil, licensePath: "\(cmlx)/mlx-c"),
    Component(name: "{fmt}", url: "https://github.com/fmtlib/fmt", version: nil, licensePath: "\(cmlx)/fmt"),
    Component(name: "JSON for Modern C++", url: "https://github.com/nlohmann/json", version: nil, licensePath: "\(cmlx)/json"),
    Component(name: "metal-cpp", url: "https://developer.apple.com/metal/cpp/", version: nil, licensePath: "\(cmlx)/metal-cpp"),
]

// Lines of at most 80 characters, like the license texts that follow.
var text = """
    Turbo MLX: acknowledgements

    Turbo MLX is free software under the MIT License, © 2026 Stefano Rinaldo:
    https://github.com/rinste/turbo-mlx

    It is built on open-source software; thank you to its authors. The models
    are not part of the app: each one is downloaded from Hugging Face under its
    own license (Settings → About).

    The engine's Swift ports follow these projects:


    """
for project in ported {
    text += "  \(project.name): \(project.use)\n    \(project.url)\n"
}
text += "\nThe app contains these libraries:\n\n"
for component in packages + bundled {
    text += "  \(component.name)\(component.version.map { " \($0)" } ?? "")\n    \(component.url)\n"
}

var apacheText: String?
var apacheUsers: [String] = []
var notices: [(String, String)] = []
var sections: [(String, String)] = []

for project in ported {
    sections.append((project.name, read("Engine/Licenses/\(project.name).txt")))
}
for component in packages + bundled {
    guard let path = file(in: component.licensePath, named: ["LICENSE", "COPYING"]) else {
        fail("no license for \(component.name) in \(component.licensePath)")
    }
    let license = read(path)
    if isApache(license) {
        // The standard text once; what a project adds after it is an extra permission.
        apacheText = apacheText ?? license.components(separatedBy: "END OF TERMS AND CONDITIONS").first.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\nEND OF TERMS AND CONDITIONS"
        }
        apacheUsers.append(component.name)
    } else {
        sections.append((component.name, license))
    }
    if let notice = file(in: component.licensePath, named: ["NOTICE"]) {
        notices.append((component.name, read(notice)))
    }
}

/// `words` joined by spaces into lines of at most 80 characters.
func wrap(_ words: [String]) -> String {
    var lines = [""]
    for word in words {
        if !lines[lines.count - 1].isEmpty, lines[lines.count - 1].count + 1 + word.count > 80 { lines.append("") }
        lines[lines.count - 1] += (lines[lines.count - 1].isEmpty ? "" : " ") + word
    }
    return lines.joined(separator: "\n")
}

let rule = String(repeating: "=", count: 80)
for (name, license) in sections {
    text += "\n\(rule)\n\(name)\n\(rule)\n\n\(license)\n"
}
if let apacheText {
    let users = wrap(("The license of " + apacheUsers.joined(separator: ", ") + ":").split(separator: " ").map(String.init))
    text += "\n\(rule)\nApache License 2.0\n\(rule)\n\n\(users)\n\n\(apacheText)\n"
}
for (name, notice) in notices {
    text += "\n\(rule)\nNOTICE of \(name)\n\(rule)\n\n\(notice)\n"
}

do {
    try text.write(toFile: output, atomically: true, encoding: .utf8)
} catch {
    fail("cannot write \(output): \(error.localizedDescription)")
}
print("Wrote \(output): \(sections.count) licenses, the Apache License for \(apacheUsers.count) components, \(notices.count) notices.")
