#!/bin/zsh
# Writes the Sparkle feed of one release, appcast.xml next to its disk image: the version and
# build number of the app inside the image, where to download it on GitHub, and the image's EdDSA
# signature (Sparkle's sign_update, with the private key in the keychain). release.sh runs it; the
# feed is attached to the GitHub release with the image, and the app reads the one of the latest
# release (SUFeedURL in TurboMLX-Info.plist).
#
# Usage: scripts/make-appcast.sh build/release/Turbo-MLX-1.1.dmg
set -euo pipefail

cd "${0:A:h}/.."
DMG="${1:?the disk image of the release}"
REPO="rinste/turbo-mlx"
OUT="${DMG:h}/appcast.xml"

# Sparkle's tools come with its package: where release.sh resolves it, else where Xcode does.
tools=(build/SourcePackages/artifacts/sparkle/Sparkle/bin/sign_update(N)
       ~/Library/Developer/Xcode/DerivedData/*/SourcePackages/artifacts/sparkle/Sparkle/bin/sign_update(N))
SIGN_UPDATE="${tools[1]:-}"
[[ -x "$SIGN_UPDATE" ]] || {
  print -u2 "Sparkle's sign_update is missing; resolve the packages first:"
  print -u2 "  xcodebuild -resolvePackageDependencies -project TurboMLX.xcodeproj -scheme TurboMLX -clonedSourcePackagesDirPath build/SourcePackages"
  exit 1
}

# What the image holds, read from the app in it.
MOUNT=$(mktemp -d)
hdiutil attach -nobrowse -readonly -mountpoint "$MOUNT" "$DMG" >/dev/null
trap 'hdiutil detach "$MOUNT" -quiet' EXIT
INFO="$MOUNT/Turbo MLX.app/Contents/Info.plist"
VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$INFO")
BUILD=$(/usr/libexec/PlistBuddy -c "Print CFBundleVersion" "$INFO")
MINIMUM=$(/usr/libexec/PlistBuddy -c "Print LSMinimumSystemVersion" "$INFO")

# sparkle:edSignature="…" length="…"
SIGNATURE=$("$SIGN_UPDATE" "$DMG")
[[ "$SIGNATURE" == *edSignature=* ]] || { print -u2 "sign_update did not sign ${DMG:t}: $SIGNATURE"; exit 1 }

cat > "$OUT" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Turbo MLX</title>
    <item>
      <title>Turbo MLX $VERSION</title>
      <pubDate>$(LC_ALL=C date -R)</pubDate>
      <sparkle:version>$BUILD</sparkle:version>
      <sparkle:shortVersionString>$VERSION</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>$MINIMUM</sparkle:minimumSystemVersion>
      <sparkle:fullReleaseNotesLink>https://github.com/$REPO/releases/tag/v$VERSION</sparkle:fullReleaseNotesLink>
      <enclosure url="https://github.com/$REPO/releases/download/v$VERSION/${DMG:t}" $SIGNATURE type="application/octet-stream"/>
    </item>
  </channel>
</rss>
EOF
print "Wrote $OUT: Turbo MLX $VERSION (build $BUILD)"
