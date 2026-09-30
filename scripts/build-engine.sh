#!/bin/zsh
# Builds the engine (Engine/Package.swift) in Release with xcodebuild, which compiles mlx-swift's
# Metal shaders (`swift build` does not on macOS), and copies the binary with its resource bundles.
# The app's build runs this (scripts/embed-engine.sh) to embed the engine in the bundle; by hand it
# gives a binary for `turbo-engine verify`.
#
# Usage: scripts/build-engine.sh                 -> build/bin/turbo-engine
#        scripts/build-engine.sh path/to/bin      -> that directory instead
#        scripts/build-engine.sh path/to/bin path/to/resources
#                                                -> the resource bundles go there instead of next to
#                                                   the binary (Contents/Resources inside an app)
set -euo pipefail

cd "${0:A:h}/.."
DEST="${1:-build/bin}"
RESOURCES="${2:-$DEST}"
DERIVED="build/engine"
PRODUCTS="$DERIVED/Build/Products/Release"
# The name of every resource bundle this build has made, one per line.
MADE="$DERIVED/resource-bundles.txt"

step() { print -P "%F{cyan}==>%f $*" }

# xcodebuild never deletes the product of a target the package has lost: when the engine left
# swift-transformers, two bundles of its packages (swift-transformers_Hub, swift-crypto_Crypto)
# stayed in the products folder and went into every app built after. So the bundles are removed
# before each build, which makes the current ones again in seconds, and their names are kept: a
# bundle the build no longer makes is taken out of wherever an earlier build installed it.
made=()
[[ -f "$MADE" ]] && made=( ${(f)"$(<"$MADE")"} )
made+=( "$PRODUCTS"/*.bundle(N:t) )
mkdir -p "$DERIVED"
print -l ${(ou)made} > "$MADE"
for bundle in "$PRODUCTS"/*.bundle(N); do
  rm -rf -- "$bundle"
done

step "Building turbo-engine (Release)"
# xcodebuild builds a package from its own folder. mlx-swift declares a package plug-in, which
# xcodebuild will not run unattended unless its validation is skipped.
(cd Engine && xcodebuild -quiet -scheme turbo-engine -configuration Release \
  -destination "platform=macOS,arch=arm64" -derivedDataPath "../$DERIVED" \
  -skipPackagePluginValidation -skipMacroValidation build)

BIN=$(find "$PRODUCTS" -maxdepth 1 -type f -name turbo-engine | head -1)
[[ -n "$BIN" ]] || { print -u2 "turbo-engine was not produced; see the xcodebuild output"; exit 1 }
bundles=( "$PRODUCTS"/*.bundle(N:t) )
made+=( $bundles )
print -l ${(ou)made} > "$MADE"

step "Installing to $DEST"
mkdir -p "$DEST" "$RESOURCES"
install -m 755 "$BIN" "$DEST/turbo-engine"
# The resource bundles sit next to the binary in the products folder, mlx-swift's Metal library
# (mlx-swift_Cmlx.bundle) among them. MLX looks for them next to the binary, then in the
# resources of the main bundle: Contents/Resources when the engine runs from inside the app.
for name in $bundles; do
  rm -rf -- "${RESOURCES:?}/$name"
  cp -R "$PRODUCTS/$name" "$RESOURCES/"
done
# Those of packages the engine no longer has, left there by an earlier install.
for name in ${made:|bundles}; do
  rm -rf -- "${RESOURCES:?}/$name"
done
print "Installed: $DEST/turbo-engine"
if [[ $# -eq 0 ]]; then
  print "Check a port with $DEST/turbo-engine verify <fixture> (Engine/README.md)."
fi
