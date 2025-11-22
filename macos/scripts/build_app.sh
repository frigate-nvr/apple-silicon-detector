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

# Change to the script directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Clear screen and show header
clear
echo "═══════════════════════════════════════════════════════"
echo "  Frigate Detector - Apple Silicon Edition"
echo "═══════════════════════════════════════════════════════"
echo ""

# Choose Python per policy:
# 1) Use `python3` if it is >= 3.11
# 2) Else try `python3.11`
# 3) Else error

use_py3() {
  python3 - <<'PYVER' 2>/dev/null
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
PYVER
}

if command -v python3 >/dev/null 2>&1 && use_py3; then
  PYBIN="python3"
elif command -v python3.11 >/dev/null 2>&1; then
  PYBIN="python3.11"
else
  echo "❌ ERROR: Python 3.11 is required."
  echo "   Please install it (e.g., 'brew install python@3.11') and try again."
  echo ""
  echo "Press any key to close this window..."
  read -n 1
  exit 1
fi

echo "✓ Using Python: $PYBIN ($($PYBIN --version 2>&1))"
echo ""

# Setup virtual environment
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  if ! "$PYBIN" -m venv venv; then
    echo "❌ ERROR: Failed to create virtual environment"
    echo ""
    echo "Press any key to close this window..."
    read -n 1
    exit 1
  fi
  echo "✓ Virtual environment created"
else
  echo "✓ Virtual environment found"
fi
echo ""

PIP="venv/bin/pip3"
PY="venv/bin/python3"

# Install dependencies
echo "Installing dependencies..."
if ! "$PIP" install --quiet --upgrade pip || ! "$PIP" install --quiet -r requirements.txt; then
  echo "❌ ERROR: Failed to install dependencies"
  echo "   Check the log file for details: $HOME/Library/Logs/FrigateDetector/FrigateDetector.log"
  echo ""
  echo "Press any key to close this window..."
  read -n 1
  exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Setup logging
LOG_DIR="$HOME/Library/Logs/FrigateDetector"
LOG_FILE="$LOG_DIR/FrigateDetector.log"
mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  Starting detector..."
echo "═══════════════════════════════════════════════════════"
echo "  Log file: $LOG_FILE"
echo "  Model: AUTO"
echo ""
echo "  (This window will stay open while the detector is running)"
echo "  (Press Ctrl+C to stop the detector)"
echo "═══════════════════════════════════════════════════════"
echo ""

# Run the detector with both console and log output
"$PY" detector/zmq_onnx_client.py --model AUTO 2>&1 | tee -a "$LOG_FILE"

# If we get here, the detector has stopped
EXIT_CODE=$?
echo ""
echo "═══════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
  echo "  Detector stopped normally"
else
  echo "  Detector stopped with error (exit code: $EXIT_CODE)"
  echo "  Check the log file for details: $LOG_FILE"
fi
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Press any key to close this window..."
read -n 1
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
    <string>FrigateDetector</string>
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
    <string>1.1.0</string>
    <key>CFBundleVersion</key>
    <string>2</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

cat > "$APP_DIR/Contents/MacOS/FrigateDetector" <<'EXEC'
#!/bin/zsh

APP_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
PROJECT_DIR="$APP_DIR/Contents/Resources/app"

# Ensure run.sh is executable
chmod +x "$PROJECT_DIR/run.sh"

# Create a .command file that Terminal can open directly
# .command files are automatically opened in Terminal when double-clicked
COMMAND_FILE="$PROJECT_DIR/FrigateDetector.command"
cat > "$COMMAND_FILE" <<'CMD'
#!/bin/zsh
cd "$(dirname "$0")"
exec ./run.sh
CMD
chmod +x "$COMMAND_FILE"

# Open the .command file using LaunchServices - this doesn't require AppleScript permissions
# The .command extension tells macOS to open it in Terminal automatically
open "$COMMAND_FILE"

# Keep the app process alive briefly to ensure Terminal launches
sleep 1
EXEC

chmod +x "$APP_DIR/Contents/MacOS/FrigateDetector"

# Attempt to remove quarantine attribute for locally built app (no network download)
# xattr doesn't support -r flag, so we use find + xargs
if command -v xattr >/dev/null 2>&1; then
  find "$APP_DIR" -type f -print0 | xargs -0 xattr -d com.apple.quarantine 2>/dev/null || true
fi

echo "App bundle created at macos/${APP_NAME}.app"


