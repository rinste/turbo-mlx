import Foundation

/// Whether a Hugging Face repository is a checkpoint Turbo MLX runs, from what the hub says of it
/// (`HubModel`): the model it was made from names the family, and its files must make a complete
/// checkpoint of that family, the way `ModelLocator.isComplete` checks one on disk.
nonisolated enum ModelCompatibility {
    nonisolated enum Verdict: Hashable, Sendable {
        /// Runs as this family (with this mflux registry key for FLUX.2 Klein).
        case supported(ModelFamily, variant: String?)
        /// The files fit one of these families, but nothing tells which: the user picks.
        case unknownFamily([ModelFamily])
        case unsupported(String)
    }

    /// The models the engine was checked against, by the `base_model` their conversions name: the
    /// family and, for FLUX.2 Klein, the registry entry. Other bases of the same names (Klein 9B KV,
    /// Qwen-Image 2.1, Qwen-Image-Edit 2509, Z-Image base) differ in ways the engine does not
    /// handle, so they are left out until they are ported.
    static let bases: [String: (family: ModelFamily, variant: String?)] = [
        "inclusionai/ming-image-0.1-design": (.ming, nil),
        "tongyi-mai/z-image-turbo": (.zImageTurbo, nil),
        "black-forest-labs/flux.2-klein-4b": (.flux2Klein, "flux2-klein-4b"),
        "black-forest-labs/flux.2-klein-9b": (.flux2Klein, "flux2-klein-9b"),
        "black-forest-labs/flux.2-klein-base-4b": (.flux2Klein, "flux2-klein-base-4b"),
        "black-forest-labs/flux.2-klein-base-9b": (.flux2Klein, "flux2-klein-base-9b"),
        "openmoss-team/moss-image-sft": (.mossImage, nil),
        "openmoss-team/moss-image-base": (.mossImage, nil),
        "qwen/qwen-image": (.qwenImage, nil),
        "qwen/qwen-image-2512": (.qwenImage, nil),
        "qwen/qwen-image-edit-2511": (.qwenImageEdit, nil),
        // mlx-serve's packs of the release, which carry DINOv3 and the VAE's encoder (the release alone does not).
        "speridlabs/idol": (.idol, nil),
    ]

    /// The families that share mflux's `transformer`, `text_encoder`, `vae` and `tokenizer` layout,
    /// which the files alone cannot tell apart.
    static let sharedLayout: [ModelFamily] = [.zImageTurbo, .flux2Klein, .qwenImage, .qwenImageEdit]

    static func assess(_ model: HubModel) -> Verdict {
        let files = Set(model.files)
        if let (family, variant) = family(of: model) {
            if let missing = missingPiece(of: family, in: files) { return .unsupported(missing) }
            return .supported(family, variant: variant)
        }
        if let base = model.baseModels.first {
            let name = base.split(separator: "/").last.map(String.init) ?? base
            return .unsupported("Made from \(name), which Turbo MLX doesn't run yet.")
        }
        // No base and no telling name: Ming-Image's folders are its own, the others share theirs.
        if missingPiece(of: .ming, in: files) == nil { return .supported(.ming, variant: nil) }
        let fitting = sharedLayout.filter { missingPiece(of: $0, in: files) == nil }
        return fitting.isEmpty ? .unsupported("Its files aren't laid out as an mflux checkpoint.") : .unknownFamily(fitting)
    }

    /// The family from the repository's base model, or else from its name when it names none.
    private static func family(of model: HubModel) -> (ModelFamily, String?)? {
        if !model.baseModels.isEmpty {
            return model.baseModels.lazy.compactMap { bases[$0.lowercased()] }.first.map { ($0.family, $0.variant) }
        }
        let name = model.id.lowercased()
        if name.contains("ming-image") { return (.ming, nil) }
        if name.contains("moss-image") { return (.mossImage, nil) }
        if name.contains("z-image-turbo") { return (.zImageTurbo, nil) }
        if name.contains("qwen-image-edit-2511") { return (.qwenImageEdit, nil) }
        if name.contains("klein"), ["flux2", "flux.2", "flux-2"].contains(where: name.contains), !name.contains("kv") {
            let size = name.contains("9b") ? "9b" : "4b"
            return (.flux2Klein, name.contains("base") ? "flux2-klein-base-\(size)" : "flux2-klein-\(size)")
        }
        return nil
    }

    /// What keeps `files` (a repository's paths) from being a complete checkpoint of `family`, or
    /// nil when nothing does: the remote side of `ModelLocator.isComplete`, which also reads the
    /// shard indexes once the files are on disk.
    static func missingPiece(of family: ModelFamily, in files: Set<String>) -> String? {
        let topLevel = files.filter { !$0.contains("/") }
        for name in family.requiredFiles {
            let regex = HubDownloader.regex(fnmatch: name)
            // A name in a sub-folder (Idol's `dinov3/model.safetensors`) is looked for among the repository's paths.
            let candidates = name.contains("/") ? files : topLevel
            guard candidates.contains(where: { regex.firstMatch(in: $0, range: NSRange($0.startIndex..., in: $0)) != nil }) else {
                return "It has no \(name)."
            }
        }
        if let tokenizer = family.tokenizerFile, !files.contains(tokenizer) {
            return "It has no tokenizer (\(tokenizer))."
        }
        for component in family.components {
            let weights = component == "."
                ? topLevel.contains { $0.hasSuffix(".safetensors") }
                : files.contains { $0.hasPrefix(component + "/") && $0.hasSuffix(".safetensors") && $0.split(separator: "/").count == 2 }
            if !weights { return component == "." ? "It has no weights." : "It has no \(component)/ weights." }
        }
        return nil
    }
}
