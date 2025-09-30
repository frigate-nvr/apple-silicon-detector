#!/bin/zsh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
APP_NAME="FrigateDetector"
APP_DIR="$PROJECT_DIR/macos/${APP_NAME}.app"
RES_DIR="$APP_DIR/Contents/Resources"
PAYLOAD_DIR="$RES_DIR/app"

echo "Recreating app bundle at $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$RES_DIR" "$PAYLOAD_DIR"

# Stage minimal payload required to run in isolation
mkdir -p "$PAYLOAD_DIR/detector"
rsync -a \
  "$PROJECT_DIR/detector/" "$PAYLOAD_DIR/detector/"

# Copy top-level files needed at runtime
cp "$PROJECT_DIR/requirements.txt" "$PAYLOAD_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/Makefile" "$PAYLOAD_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/README.md" "$PAYLOAD_DIR/" 2>/dev/null || true

# Create embedded runner (used by applet)
cat > "$PAYLOAD_DIR/run.sh" <<'RUNNER'
#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PYBIN="python3.11"
command -v "$PYBIN" >/dev/null 2>&1 || PYBIN="python3"

if [ ! -d "venv" ]; then
  "$PYBIN" -m venv venv
fi

PIP="venv/bin/pip3"
PY="venv/bin/python3"

"$PIP" install --upgrade pip
"$PIP" install -r requirements.txt

"$PY" detector/zmq_onnx_client.py \
  --model AUTO
RUNNER
chmod +x "$PAYLOAD_DIR/run.sh"

# If an icon exists at macos/AppIcon.icns, copy it in
if [ -f "$PROJECT_DIR/macos/AppIcon.icns" ]; then
  cp "$PROJECT_DIR/macos/AppIcon.icns" "$RES_DIR/AppIcon.icns"
fi

cat > "$APP_DIR/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>applet</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.local.FrigateDetector</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>FrigateDetector</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

cat > "$APP_DIR/Contents/MacOS/applet" <<'APPLET'
#!/bin/zsh

APP_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
PROJECT_DIR="$APP_DIR/Contents/Resources/app"
ESCAPED_DIR=$(printf "%q" "$PROJECT_DIR")
CMD="cd $ESCAPED_DIR; chmod +x ./run.sh; ./run.sh"

osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "$CMD"
end tell
APPLESCRIPT
APPLET

chmod +x "$APP_DIR/Contents/MacOS/applet"

# Attempt to remove quarantine attribute for locally built app (no network download)
if command -v xattr >/dev/null 2>&1; then
  if xattr -h 2>&1 | grep -q "-r"; then
    xattr -r -d com.apple.quarantine "$APP_DIR" || true
  else
    find "$APP_DIR" -type f -print0 | xargs -0 xattr -d com.apple.quarantine 2>/dev/null || true
  fi
fi

echo "App bundle created at macos/${APP_NAME}.app"


