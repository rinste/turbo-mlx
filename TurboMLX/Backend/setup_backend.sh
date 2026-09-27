#!/bin/zsh
# Creates (or repairs) the Python environment Turbo MLX uses to run mflux.
#
# Usage: setup_backend.sh <venv-dir> <mflux-requirement>
#
# Environment, set by the app:
#   TURBO_UV               uv binary bundled with the app
#   UV_PYTHON_INSTALL_DIR  where uv puts the private Python it downloads
#   UV_CACHE_DIR           uv's package cache
#   TURBO_REINSTALL=1      start from an empty environment
#
# With the bundled uv the Mac needs neither Python nor the developer tools: uv downloads a
# standalone Python and every package. A uv or Python 3.10–3.13 found on the system is only a
# fallback for development builds.
set -euo pipefail

VENV_DIR="$1"
MFLUX_REQ="$2"
PYTHON_VERSION="${TURBO_PYTHON_VERSION:-3.12}"

# Lines starting with "==> " are shown to the user as the current step.
log() { print -r -- "==> $*" }

find_uv() {
  local c
  for c in "${TURBO_UV:-}" "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" \
           /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [[ -n "$c" && -x "$c" ]]; then print -r -- "$c"; return 0; fi
  done
  return 1
}

find_python() {
  local c
  for c in "$(command -v python3 2>/dev/null || true)" /opt/homebrew/bin/python3 /usr/local/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/Current/bin/python3; do
    [[ -n "$c" && -x "$c" ]] || continue
    if "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      print -r -- "$c"; return 0
    fi
  done
  return 1
}

mkdir -p "${VENV_DIR:h}"

if [[ "${TURBO_REINSTALL:-0}" == 1 ]]; then
  log "Removing the previous environment"
  rm -rf "$VENV_DIR"
elif [[ -e "$VENV_DIR" ]] && ! "$VENV_DIR/bin/python" -c 'import sys' 2>/dev/null; then
  log "The existing environment is broken: recreating it"
  rm -rf "$VENV_DIR"
fi

if UV="$(find_uv)"; then
  print -r -- "uv: $UV ($("$UV" --version))"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Setting up Python $PYTHON_VERSION"
    "$UV" venv --python "$PYTHON_VERSION" --managed-python "$VENV_DIR" \
      || "$UV" venv --python "$PYTHON_VERSION" "$VENV_DIR"
  fi
  log "Installing mflux and its dependencies (about 1 GB)"
  "$UV" pip install --python "$VENV_DIR/bin/python" --upgrade "$MFLUX_REQ"
elif PY="$(find_python)"; then
  print -r -- "Python: $PY (uv not found, using pip: this is slower)"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Setting up the Python environment"
    "$PY" -m venv "$VENV_DIR"
  fi
  log "Installing mflux and its dependencies (about 1 GB)"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install --upgrade "$MFLUX_REQ"
else
  print -u2 -r -- "Found neither uv nor a Python 3.10–3.13 to create the environment."
  exit 1
fi

log "Checking the installation"
"$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import version

import mlx.core as mx
from mflux.models.flux2.variants import Flux2Klein  # noqa: F401
from mflux.models.ming_image import MingImage  # noqa: F401  (fails if this mflux lacks Ming-Image)
from mflux.models.z_image.variants.z_image import ZImage  # noqa: F401

print(f"mflux {version('mflux')} · mlx {mx.__version__} · Metal: {mx.metal.is_available()}")
PY
# The app compares this with the mflux it expects, to update the engine after an app update.
print -r -- "$MFLUX_REQ" > "$VENV_DIR/.turbo-mflux-requirement"
log "Engine ready"
