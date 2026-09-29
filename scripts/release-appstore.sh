#!/bin/zsh
# Builds Turbo MLX for the Mac App Store: the "TurboMLX App Store" target (no Sparkle, the sandbox
# without temporary exceptions, models in the container unless the user picks a folder), archived
# and exported as the installer package App Store Connect takes, and uploaded with --upload. The
# version and build number are the project's, as in the GitHub channel (scripts/release.sh):
# App Store Connect refuses a build number it has seen for the same version.
#
# Needed once, in the Apple Developer account and App Store Connect, by the account holder:
#   - the certificates "Apple Distribution" and "Mac Installer Distribution" in this Mac's
#     keychain (Xcode → Settings → Accounts → Manage Certificates → +);
#   - the explicit App ID io.github.rinste.TurboMLX (Certificates, Identifiers & Profiles) and a
#     "Mac App Store Connect" provisioning profile for it, downloaded and opened once; its name
#     goes in APPSTORE_PROFILE;
#   - the app's record in App Store Connect (Apps → + → New App, macOS, that bundle ID);
#   - for --upload, an App Store Connect API key (Users and Access → Integrations → Team Keys,
#     role Developer or above): its AuthKey_<key id>.p8 in ~/.appstoreconnect/private_keys/, and
#     the key id and the issuer id in APPSTORE_KEY_ID and APPSTORE_ISSUER.
#
# Usage: APPSTORE_PROFILE="<profile name>" scripts/release-appstore.sh
#          -> build/appstore/Turbo MLX.pkg, checked but not sent
#        APPSTORE_PROFILE=… APPSTORE_KEY_ID=… APPSTORE_ISSUER=… scripts/release-appstore.sh --upload
#          -> the same, then uploaded: it shows up in App Store Connect (TestFlight) after
#             Apple's processing, ready to test and to submit for review.
set -euo pipefail

cd "${0:A:h}/.."
TEAM_ID="${TEAM_ID:-$(security find-identity -v -p codesigning \
  | sed -nE 's/.*"Apple Distribution: .*\(([A-Z0-9]{10})\)"$/\1/p' | head -1)}"
[[ -n "$TEAM_ID" ]] || { print -u2 "No Apple Distribution certificate in the keychain (see the header)."; exit 1 }
[[ -n "${APPSTORE_PROFILE:-}" ]] || { print -u2 "Name the Mac App Store provisioning profile in APPSTORE_PROFILE."; exit 1 }
if [[ "${1:-}" == --upload && ( -z "${APPSTORE_KEY_ID:-}" || -z "${APPSTORE_ISSUER:-}" ) ]]; then
  print -u2 "--upload needs APPSTORE_KEY_ID and APPSTORE_ISSUER (see the header)."
  exit 1
fi
BUNDLE_ID="io.github.rinste.TurboMLX"
OUT="build/appstore"
ARCHIVE="$OUT/TurboMLX-AppStore.xcarchive"

step() { print -P "%F{cyan}==>%f $*" }

rm -rf "$OUT"
mkdir -p "$OUT"

step "Archiving (Release, Apple Distribution)"
xcodebuild -quiet -project TurboMLX.xcodeproj -scheme "TurboMLX App Store" -configuration Release \
  -destination "generic/platform=macOS" -archivePath "$ARCHIVE" -clonedSourcePackagesDirPath build/SourcePackages archive \
  CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="Apple Distribution" DEVELOPMENT_TEAM="$TEAM_ID" \
  PROVISIONING_PROFILE_SPECIFIER="$APPSTORE_PROFILE" OTHER_CODE_SIGN_FLAGS="--timestamp"

step "Checking the archive"
APP="$ARCHIVE/Products/Applications/Turbo MLX.app"
[[ ! -e "$APP/Contents/Frameworks/Sparkle.framework" ]] || { print -u2 "Sparkle is in the App Store build."; exit 1 }
if codesign -d --entitlements - "$APP" 2>/dev/null | grep -q "temporary-exception"; then
  print -u2 "The App Store build has a temporary-exception entitlement."
  exit 1
fi
[[ -x "$APP/Contents/MacOS/turbo-engine" ]] || { print -u2 "turbo-engine is missing from the app."; exit 1 }
codesign -d --entitlements - "$APP/Contents/MacOS/turbo-engine" 2>/dev/null | grep -q "com.apple.security.inherit" || {
  print -u2 "turbo-engine does not inherit the sandbox (Engine/turbo-engine.entitlements)."
  exit 1
}
[[ -f "$APP/Contents/Resources/PrivacyInfo.xcprivacy" ]] || { print -u2 "The privacy manifest is missing."; exit 1 }

step "Exporting the installer package"
cat > "$OUT/ExportOptions.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key>
	<string>app-store-connect</string>
	<key>destination</key>
	<string>export</string>
	<key>teamID</key>
	<string>$TEAM_ID</string>
	<key>signingStyle</key>
	<string>manual</string>
	<key>signingCertificate</key>
	<string>Apple Distribution</string>
	<key>installerSigningCertificate</key>
	<string>Mac Installer Distribution</string>
	<key>provisioningProfiles</key>
	<dict>
		<key>$BUNDLE_ID</key>
		<string>$APPSTORE_PROFILE</string>
	</dict>
</dict>
</plist>
EOF
xcodebuild -quiet -exportArchive -archivePath "$ARCHIVE" -exportPath "$OUT" -exportOptionsPlist "$OUT/ExportOptions.plist"
PKG=$(ls "$OUT"/*.pkg | head -1)
VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist")
BUILD=$(/usr/libexec/PlistBuddy -c "Print CFBundleVersion" "$APP/Contents/Info.plist")
print "Package: $PKG (version $VERSION, build $BUILD)"

if [[ "${1:-}" == --upload ]]; then
  step "Uploading to App Store Connect"
  xcrun altool --upload-app --type macos --file "$PKG" --apiKey "$APPSTORE_KEY_ID" --apiIssuer "$APPSTORE_ISSUER"
  step "Uploaded: it appears in App Store Connect → TestFlight once Apple has processed it"
else
  step "Done: $PKG (not sent; run again with --upload to send it)"
fi
