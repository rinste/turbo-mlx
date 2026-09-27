#!/bin/zsh
# Builds Turbo MLX for distribution outside the Mac App Store: Developer ID signature,
# notarization, stapling and a disk image.
#
# One-time setup (stores an app-specific password in the keychain):
#   xcrun notarytool store-credentials turbo-mlx --apple-id <apple-id> --team-id YOUR_TEAM_ID
#
# Usage: scripts/release.sh            -> build/release/Turbo-MLX-<version>.dmg
#        NOTARY_PROFILE=other scripts/release.sh
set -euo pipefail

cd "${0:A:h}/.."
TEAM_ID="YOUR_TEAM_ID"
PROFILE="${NOTARY_PROFILE:-turbo-mlx}"
OUT="build/release"
ARCHIVE="$OUT/TurboMLX.xcarchive"

step() { print -P "%F{cyan}==>%f $*" }

rm -rf "$OUT"
mkdir -p "$OUT"

step "Archiving (Release, Developer ID)"
xcodebuild -quiet -project TurboMLX.xcodeproj -scheme TurboMLX -configuration Release \
  -destination "generic/platform=macOS" -archivePath "$ARCHIVE" archive \
  CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="Developer ID Application" DEVELOPMENT_TEAM="$TEAM_ID" \
  OTHER_CODE_SIGN_FLAGS="--timestamp"

step "Exporting"
xcodebuild -quiet -exportArchive -archivePath "$ARCHIVE" -exportPath "$OUT/export" \
  -exportOptionsPlist scripts/ExportOptions.plist
APP="$OUT/export/Turbo MLX.app"
VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist")

step "Verifying the signature"
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -dvv "$APP/Contents/MacOS/uv" 2>&1 | grep -E "Authority=Developer ID|flags"

step "Notarizing (profile $PROFILE)"
ditto -c -k --keepParent "$APP" "$OUT/notarize.zip"
xcrun notarytool submit "$OUT/notarize.zip" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$APP"
spctl --assess --type execute --verbose "$APP"

step "Disk image"
DMG="$OUT/Turbo-MLX-$VERSION.dmg"
STAGING="$OUT/dmg"
mkdir -p "$STAGING"
ditto "$APP" "$STAGING/Turbo MLX.app"
ln -s /Applications "$STAGING/Applications"
hdiutil create -volname "Turbo MLX" -srcfolder "$STAGING" -ov -format UDZO "$DMG" >/dev/null
codesign --sign "Developer ID Application" --timestamp "$DMG"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$DMG"

step "Done: $DMG"
