#!/bin/zsh
# Builds Turbo MLX for distribution outside the Mac App Store: Developer ID signature,
# notarization, stapling and a disk image. It needs a "Developer ID Application" certificate in
# the keychain, whose team signs the app, and the notarization credentials, stored once: an
# app-specific password of the team's Apple ID (account.apple.com → Sign-In and Security), saved
# by the command below, which asks for the Apple ID and the password.
#   xcrun notarytool store-credentials turbo-mlx --team-id <team-id>
#
# Usage: scripts/release.sh            -> build/release/Turbo-MLX-<version>.dmg
#        NOTARY_PROFILE=other scripts/release.sh
#        TEAM_ID=<team-id> scripts/release.sh   (with more than one Developer ID certificate)
#
# To publish a version: raise MARKETING_VERSION (and CURRENT_PROJECT_VERSION) in the project, run
# `swift scripts/make-acknowledgements.swift` if the engine's dependencies changed, run this
# script and attach the DMG to a GitHub release tagged with the same number (v1.1 for 1.1), not
# a pre-release: that is the release the app's update check compares itself with.
set -euo pipefail

cd "${0:A:h}/.."
# The team of the Developer ID certificate in the keychain, unless TEAM_ID names one.
TEAM_ID="${TEAM_ID:-$(security find-identity -v -p codesigning \
  | sed -nE 's/.*"Developer ID Application: .*\(([A-Z0-9]{10})\)"$/\1/p' | head -1)}"
[[ -n "$TEAM_ID" ]] || { print -u2 "No Developer ID Application certificate in the keychain."; exit 1 }
PROFILE="${NOTARY_PROFILE:-turbo-mlx}"
OUT="build/release"
ARCHIVE="$OUT/TurboMLX.xcarchive"

step() { print -P "%F{cyan}==>%f $*" }

# Sends a file to Apple's notary service and waits; when Apple does not accept it, prints the
# service's log, which names each file it objects to and why, and stops.
notarize() {
  local result id notary_status
  result=$(xcrun notarytool submit "$1" --keychain-profile "$PROFILE" --wait --output-format json) || true
  id=$(print -r -- "$result" | plutil -extract id raw -o - - 2>/dev/null) || id=""
  notary_status=$(print -r -- "$result" | plutil -extract status raw -o - - 2>/dev/null) || notary_status=""
  print "Notarization of ${1:t}: ${notary_status:-no answer} ${id:+(submission $id)}"
  if [[ "$notary_status" != Accepted ]]; then
    [[ -n "$id" ]] && xcrun notarytool log "$id" --keychain-profile "$PROFILE" || print -u2 -r -- "$result"
    exit 1
  fi
}

# The credentials first, so that a missing profile does not stop the script after the build.
xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null || {
  print -u2 "No working notarization profile \"$PROFILE\". Store it once with:"
  print -u2 "  xcrun notarytool store-credentials $PROFILE --team-id $TEAM_ID"
  exit 1
}

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
# The engine is embedded by the app's build phase (scripts/embed-engine.sh), which fails the build
# without it; checked again here, as the app cannot generate an image without it.
[[ -x "$APP/Contents/MacOS/turbo-engine" ]] || {
  print -u2 "turbo-engine is missing from the app: the Embed turbo-engine phase failed, see the archive log"
  exit 1
}
codesign -dvv "$APP/Contents/MacOS/turbo-engine" 2>&1 | grep -E "Authority=Developer ID|flags"
# The app is sandboxed: it can start the engine only if the engine inherits its sandbox.
codesign -d --entitlements - "$APP/Contents/MacOS/turbo-engine" 2>/dev/null | grep -q "com.apple.security.inherit" || {
  print -u2 "turbo-engine is not signed with Engine/turbo-engine.entitlements: the sandboxed app could not start it"
  exit 1
}

step "Notarizing (profile $PROFILE)"
ditto -c -k --keepParent "$APP" "$OUT/notarize.zip"
notarize "$OUT/notarize.zip"
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
notarize "$DMG"
xcrun stapler staple "$DMG"

step "Done: $DMG"
