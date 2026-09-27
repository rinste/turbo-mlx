#!/bin/zsh
# Builds the native engine (Engine/Package.swift) in Release with xcodebuild, which compiles
# mlx-swift's Metal shaders (`swift build` does not on macOS), and installs the binary where the
# app looks for it.
#
# Usage: scripts/build-engine.sh               -> ~/Library/Application Support/TurboMLX/bin/turbo-engine
#        scripts/build-engine.sh path/to/dir   -> that directory instead
set -euo pipefail

cd "${0:A:h}/.."
DEST="${1:-$HOME/Library/Application Support/TurboMLX/bin}"
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
mkdir -p "$DEST"
install -m 755 "$BIN" "$DEST/turbo-engine"
# The resource bundles sit next to the binary in the products folder, mlx-swift's Metal library
# (mlx-swift_Cmlx.bundle) among them; the binary finds them next to itself.
for lib in "$DERIVED"/Build/Products/Release/*.bundle(N); do
  rm -rf "$DEST/${lib:t}"
  cp -R "$lib" "$DEST/"
done
print "Installed: $DEST/turbo-engine"
print "The app picks it up on its next launch (Settings → Engine shows the path)."
