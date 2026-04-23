#!/bin/zsh
set -euo pipefail

echo "Building standalone detector CLI..."
rm -rf build

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv run)
elif [ -f ~/.local/bin/uv ]; then
    UV_CMD=(~/.local/bin/uv run)
else
    UV_CMD=()
fi

"${UV_CMD[@]}" pyinstaller pyinstaller.spec --distpath build/ \
    --workpath build/pyinstaller --noconfirm

# Cleanup intermediates
rm -rf build/pyinstaller

echo "✓ CLI build complete: build/detector/"
echo "  Run with: ./build/detector/detector --help"
