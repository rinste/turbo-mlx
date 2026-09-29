#!/bin/zsh
# Run by the app's "Embed turbo-engine" build phase: builds the engine (scripts/build-engine.sh)
# and puts it in the app bundle, signed like the app. The binary goes to Contents/MacOS, where the
# app looks for an auxiliary executable, and its resource bundles (mlx-swift's Metal library) to
# Contents/Resources, where MLX finds them from inside an app. The engine is signed with
# Engine/turbo-engine.entitlements, which make it inherit the app's sandbox: a sandboxed app
# can only start a helper that does.
#
# The app cannot generate images without it, so a failure here fails the build. The engine is
# rebuilt only when Engine/ (its dependencies included) or build-engine.sh changed since the copy
# in the bundle; the first build takes a few minutes (MLX's C++ core and Metal kernels), the next
# ones seconds.
set -uo pipefail

cd "$SRCROOT"
BIN_DIR="$TARGET_BUILD_DIR/$EXECUTABLE_FOLDER_PATH"
RES_DIR="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH"
ENGINE="$BIN_DIR/turbo-engine"
ENTITLEMENTS="$SRCROOT/Engine/turbo-engine.entitlements"

# Its sources, the resolved dependencies (a new mlx-swift must not ship an old engine) and how it
# is built.
if [[ -x "$ENGINE" && -z "$(find Engine/Package.swift Engine/Package.resolved Engine/Sources "$ENTITLEMENTS" \
      scripts/build-engine.sh -newer "$ENGINE" | head -1)" ]]; then
  print "turbo-engine is up to date in $WRAPPER_NAME"
  exit 0
fi

# The engine's own xcodebuild must not inherit this build's settings (SDKROOT, ARCHS,
# TOOLCHAINS…), which a build phase gets in its environment; it keeps only what finds Xcode.
if ! env -i HOME="$HOME" USER="${USER:-}" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
      ${DEVELOPER_DIR:+DEVELOPER_DIR="$DEVELOPER_DIR"} \
      /bin/zsh scripts/build-engine.sh "$BIN_DIR" "$RES_DIR"; then
  print "error: turbo-engine did not build; its xcodebuild output is above"
  rm -f "$ENGINE"
  exit 1
fi

# Signed with the app's identity and options: ad hoc ("-") in a local build, Developer ID with
# the hardened runtime and a timestamp in a release, so notarization accepts the nested binary.
if [[ "${CODE_SIGNING_ALLOWED:-NO}" == YES && -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]]; then
  flags=(--force --sign "$EXPANDED_CODE_SIGN_IDENTITY")
  [[ "${ENABLE_HARDENED_RUNTIME:-NO}" == YES ]] && flags+=(--options runtime)
  # In the sealed build (TURBO_MLX_CHANNEL in the "TurboMLX Sealed" target) the engine gets an
  # identifier under the app's, as Apple wants for a helper tool in a sealed app; the GitHub
  # build keeps the file name, as it always had.
  engine_id=()
  [[ "${TURBO_MLX_CHANNEL:-}" == sealed ]] && engine_id=(--identifier "$PRODUCT_BUNDLE_IDENTIFIER.turbo-engine")
  [[ -n "${OTHER_CODE_SIGN_FLAGS:-}" ]] && flags+=(${=OTHER_CODE_SIGN_FLAGS})
  for item in "$RES_DIR"/*.bundle(N) "$ENGINE"; do
    extra=()
    [[ "$item" == "$ENGINE" ]] && extra=(--entitlements "$ENTITLEMENTS" "${engine_id[@]}")
    codesign "${flags[@]}" "${extra[@]}" "$item" || {
      print "error: could not sign ${item:t}"
      rm -f "$ENGINE"
      exit 1
    }
  done
fi
print "Embedded turbo-engine in $WRAPPER_NAME"
