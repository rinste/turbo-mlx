#!/bin/zsh
# Builds the engine (Engine/Package.swift) in Release with xcodebuild, which compiles mlx-swift's
# Metal shaders (`swift build` does not on macOS), and copies the binary with its resource bundles.
# The app's build runs this (scripts/embed-engine.sh) to embed the engine in the bundle; by hand it
# gives a binary for `turbo-engine verify`, or for the app's TURBO_ENGINE.
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

step() { print -P "%F{cyan}==>%f $*" }

step "Building turbo-engine (Release)"
# xcodebuild builds a package from its own folder. mlx-swift declares a package plug-in, which
# xcodebuild will not run unattended unless its validation is skipped.
(cd Engine && xcodebuild -quiet -scheme turbo-engine -configuration Release \
  -destination "platform=macOS,arch=arm64" -derivedDataPath "../$DERIVED" \
  -skipPackagePluginValidation -skipMacroValidation build)

BIN=$(find "$DERIVED/Build/Products/Release" -maxdepth 1 -type f -name turbo-engine | head -1)
[[ -n "$BIN" ]] || { print -u2 "turbo-engine was not produced; see the xcodebuild output"; exit 1 }

step "Installing to $DEST"
mkdir -p "$DEST" "$RESOURCES"
install -m 755 "$BIN" "$DEST/turbo-engine"
# The resource bundles sit next to the binary in the products folder, mlx-swift's Metal library
# (mlx-swift_Cmlx.bundle) among them. MLX looks for them next to the binary, then in the
# resources of the main bundle: Contents/Resources when the engine runs from inside the app.
for lib in "$DERIVED"/Build/Products/Release/*.bundle(N); do
  rm -rf "$RESOURCES/${lib:t}"
  cp -R "$lib" "$RESOURCES/"
done
print "Installed: $DEST/turbo-engine"
if [[ $# -eq 0 ]]; then
  print "Check a port with $DEST/turbo-engine verify <fixture>, or run the app on it with TURBO_ENGINE=$PWD/$DEST/turbo-engine."
fi
