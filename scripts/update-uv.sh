#!/bin/zsh
# Replaces Vendor/uv/uv with a uv release for Apple Silicon, verifying its SHA-256.
#
# Usage: scripts/update-uv.sh            (latest release)
#        scripts/update-uv.sh 0.12.19    (a specific version)
set -euo pipefail

cd "${0:A:h}/.."
VERSION="${1:-$(curl -sSL -o /dev/null -w '%{url_effective}' https://github.com/astral-sh/uv/releases/latest | sed 's#.*/tag/##')}"
BASE="https://github.com/astral-sh/uv/releases/download/$VERSION"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -sSLf "$BASE/uv-aarch64-apple-darwin.tar.gz" -o "$TMP/uv.tar.gz"
EXPECTED=$(curl -sSLf "$BASE/uv-aarch64-apple-darwin.tar.gz.sha256" | awk '{print $1}')
ACTUAL=$(shasum -a 256 "$TMP/uv.tar.gz" | awk '{print $1}')
[[ "$EXPECTED" == "$ACTUAL" ]] || { print -u2 "Checksum mismatch: expected $EXPECTED, got $ACTUAL"; exit 1 }

tar -xzf "$TMP/uv.tar.gz" -C "$TMP"
install -m 755 "$TMP/uv-aarch64-apple-darwin/uv" Vendor/uv/uv
print -r -- "$VERSION" > Vendor/uv/VERSION
curl -sSLf "https://raw.githubusercontent.com/astral-sh/uv/$VERSION/LICENSE-MIT" -o TurboMLX/Resources/Licenses/uv-LICENSE.txt
print "uv $VERSION ($(Vendor/uv/uv --version))"
