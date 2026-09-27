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
xcodebuild -quiet -scheme turbo-engine -configuration Release -destination "platform=macOS" \
  -derivedDataPath "$DERIVED" -workspace Engine/.swiftpm/xcode/package.xcworkspace build 2>/dev/null \
  || xcodebuild -quiet -scheme turbo-engine -configuration Release -destination "platform=macOS" \
       -derivedDataPath "$DERIVED" build

BIN=$(find "$DERIVED/Build/Products/Release" -maxdepth 1 -type f -name turbo-engine | head -1)
[[ -n "$BIN" ]] || { print -u2 "turbo-engine was not produced; see the xcodebuild output"; exit 1 }

step "Installing to $DEST"
mkdir -p "$DEST"
install -m 755 "$BIN" "$DEST/turbo-engine"
# The metallib mlx-swift builds sits next to the binary in the products folder; keep it with it.
for lib in "$DERIVED"/Build/Products/Release/*.metallib "$DERIVED"/Build/Products/Release/*.bundle; do
  [[ -e "$lib" ]] && cp -R "$lib" "$DEST/"
done
print "Installed: $DEST/turbo-engine"
print "The app picks it up on its next launch (Settings → Engine shows the path)."
