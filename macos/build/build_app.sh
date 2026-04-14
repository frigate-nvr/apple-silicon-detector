#!/bin/zsh
set -euo pipefail

# ── Step 1: Build Swift UI ──────────────────────────────────
# Compiles the native macOS menu bar dashboard using the Swift package.
echo "Building Swift menu bar app..."
rm -rf macos/swift/.build
cd macos/swift
swift build -c release
SWIFT_BIN=".build/release/FrigateDetector"
cd ../..

# ── Step 2: Build Python Service ─────────────────────────────
# Packages the detector logic, CLI, and model utilities into a standalone
# directory structure using PyInstaller.
rm -rf build
if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv run)
elif [ -f ~/.local/bin/uv ]; then
    UV_CMD=(~/.local/bin/uv run)
else
    UV_CMD=()
fi
echo "Building Unified Python binary..."
cd macos/build
"${UV_CMD[@]}" pyinstaller ../../pyinstaller.spec --distpath ../../build/ \
    --workpath ../../build/pyinstaller --noconfirm
cd ../..

# ── Step 3: Assemble .app Bundle ─────────────────────────────
# Standard macOS application structure:
# - Contents/MacOS/        -> Native binaries and entry point scripts
# - Contents/Resources/    -> Assets, icons, and non-Mach-O runtime support (Python)
echo "Assembling FrigateDetector.app..."
APP="build/FrigateDetector.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources/detector-bin"

# Install the native Swift executable
cp "macos/swift/$SWIFT_BIN" "$APP/Contents/MacOS/FrigateDetector"
chmod +x "$APP/Contents/MacOS/FrigateDetector"

# Install the Python runtime into Resources/
# Putting the full Python 'onedir' here avoids strict codesign enforcement 
# on internal scripts and nested frameworks that are common in PyInstaller outputs.
cp -r build/detector/* "$APP/Contents/Resources/detector-bin/"

# Create a shim for the Python CLI in Contents/MacOS/
# This makes 'detector' available in the standard search path within the bundle,
# while the actual implementation lives in the Resources/ partition.
cat > "$APP/Contents/MacOS/detector" <<'EOF'
#!/bin/zsh
REAL_DIR="${0:A:h}"
EXEC_PATH="$REAL_DIR/../Resources/detector-bin/detector"
exec "$EXEC_PATH" "$@"
EOF
chmod +x "$APP/Contents/MacOS/detector"

# Info.plist and App icon
cp macos/swift/Sources/FrigateDetector/Resources/Info.plist "$APP/Contents/"
cp macos/assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

# Compiled Asset Catalog
echo "Compiling Asset Catalog..."
xcrun actool macos/swift/Sources/FrigateDetector/Resources/Assets.xcassets \
    --compile "$APP/Contents/Resources" \
    --platform macosx \
    --minimum-deployment-target 13.0 \
    --output-format human-readable-text

# ── Step 4: Codesign ─────────────────────────────────────────
# Applies an ad-hoc signature to the entire bundle.
# Using --deep ensures all nested libraries and executables are sealed correctly,
# satisfying security requirements on macOS 13+.
echo "Signing FrigateDetector.app..."
ENTITLEMENTS="macos/Entitlements.plist"

# Remove existing signatures and sign deeply
find "$APP" -type f -exec codesign --remove-signature {} \; 2>/dev/null || true
codesign --force --deep --sign - --entitlements "$ENTITLEMENTS" "$APP"

# ── Step 5: Cleanup intermediates ────────────────────────────
rm -rf build/detector build/pyinstaller

# ── Step 6: Package DMG ──────────────────────────────────────
echo "Creating DMG..."
if ! command -v create-dmg &> /dev/null; then
    echo "create-dmg could not be found. Skipping DMG creation."
else
    rm -rf build/dmg_staging
    mkdir -p build/dmg_staging
    cp -r "$APP" build/dmg_staging/

    rm -f build/FrigateDetector.dmg
    create-dmg \
      --volname "Install Frigate Detector" \
      --volicon "macos/assets/AppIcon.icns" \
      --background "macos/assets/dmg_background.png" \
      --window-pos 200 120 \
      --window-size 480 540 \
      --icon-size 100 \
      --icon "FrigateDetector.app" 240 160 \
      --hide-extension "FrigateDetector.app" \
      --app-drop-link 240 400 \
      "build/FrigateDetector.dmg" \
      "build/dmg_staging/"

    rm -rf build/dmg_staging
fi

echo "✓ Build complete!"
