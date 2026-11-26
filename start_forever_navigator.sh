#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

INSTALL_ONLY=0
FORCE_UPDATE=0
APP_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --install-only) INSTALL_ONLY=1 ;;
        --update) FORCE_UPDATE=1 ;;
        *) APP_ARGS+=("$arg") ;;
    esac
done

NEEDS_INSTALL=0
if [ "$FORCE_UPDATE" -eq 1 ] || [ ! -x ".venv/bin/python" ]; then
    NEEDS_INSTALL=1
elif ! .venv/bin/python -c "import tkinter, numpy, PIL, cv2, mss, onnxruntime, moderngl, pyrr, websockets, PySide6" >/dev/null 2>&1; then
    NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
    if command -v uv >/dev/null 2>&1; then
        if [ ! -x ".venv/bin/python" ]; then
            uv venv --python 3.12 .venv || uv venv --python 3.11 .venv || uv venv .venv
        fi
        uv pip install --python .venv/bin/python -r requirements-runtime.txt
    else
        if [ ! -x ".venv/bin/python" ]; then
            python3 -m venv .venv
        fi
        .venv/bin/python -m pip install --upgrade pip
        .venv/bin/python -m pip install -r requirements-runtime.txt
    fi
fi

if [ "$INSTALL_ONLY" -eq 1 ]; then
    echo "All ForeverNavigator runtime dependencies are installed."
    exit 0
fi

exec .venv/bin/python -m src.ui.app "${APP_ARGS[@]}"
