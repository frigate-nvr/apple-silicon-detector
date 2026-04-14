#!/bin/zsh

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_png> <output_icns>"
  exit 1
fi

INPUT_PNG="$1"
OUTPUT_ICNS="$2"

TMPDIR=$(mktemp -d)
ICONSET="$TMPDIR/icon.iconset"
mkdir -p "$ICONSET"

# Requires sips (preinstalled on macOS)
sips -z 16 16     "$INPUT_PNG" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32     "$INPUT_PNG" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32     "$INPUT_PNG" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64     "$INPUT_PNG" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128   "$INPUT_PNG" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256   "$INPUT_PNG" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256   "$INPUT_PNG" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512   "$INPUT_PNG" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512   "$INPUT_PNG" --out "$ICONSET/icon_512x512.png" >/dev/null
cp "$INPUT_PNG" "$ICONSET/icon_512x512@2x.png"

iconutil -c icns "$ICONSET" -o "$OUTPUT_ICNS"
rm -rf "$TMPDIR"

echo "Wrote $OUTPUT_ICNS"


