#!/bin/bash

# Apple Silicon Frigate Detector Installer
# This script installs the detector with desktop integration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/.local/share/apple-silicon-frigate-detector"
VENV_DIR="$INSTALL_DIR/venv"
DESKTOP_DIR="$HOME/Desktop"
APPLICATIONS_DIR="$HOME/Applications"
PACKAGE_NAME="apple-silicon-frigate-detector"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This installer is designed for macOS only"
        exit 1
    fi
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        log_info "Please install Python 3.10 or later from https://python.org"
        exit 1
    fi
    
    # Check Python version is 3.10+
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version 3.10" | awk '{print ($1 >= $2)}') == 0 ]]; then
        log_error "Python 3.10 or later is required (found: $python_version)"
        exit 1
    fi
    
    log_success "Python $python_version found"
}

# Create installation directory
create_install_dir() {
    log_info "Creating installation directory..."
    mkdir -p "$INSTALL_DIR"
    log_success "Created $INSTALL_DIR"
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment..."
    
    if [[ -d "$VENV_DIR" ]]; then
        log_warning "Virtual environment already exists, removing..."
        rm -rf "$VENV_DIR"
    fi
    
    python3 -m venv "$VENV_DIR"
    log_success "Virtual environment created"
}

# Install package
install_package() {
    log_info "Installing Apple Silicon Frigate Detector..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install from current directory (development) or PyPI (production)
    if [[ -f "pyproject.toml" ]]; then
        log_info "Installing from local source..."
        pip install -e ".[gui]"
    else
        log_info "Installing from PyPI..."
        pip install "${PACKAGE_NAME}[gui]"
    fi
    
    log_success "Package installed successfully"
}

# Create desktop shortcuts
create_desktop_shortcuts() {
    log_info "Creating desktop shortcuts..."
    
    # Create CLI shortcut
    cat > "$DESKTOP_DIR/Frigate Detector (CLI).command" << EOF
#!/bin/bash
cd "\$HOME"
source "$VENV_DIR/bin/activate"
frigate-detector --help
echo ""
echo "Usage: frigate-detector --model /path/to/your/model.onnx"
echo "Press any key to exit..."
read -n 1
EOF
    chmod +x "$DESKTOP_DIR/Frigate Detector (CLI).command"
    
    # Create GUI shortcut
    cat > "$DESKTOP_DIR/Frigate Detector (GUI).command" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
frigate-detector-gui
EOF
    chmod +x "$DESKTOP_DIR/Frigate Detector (GUI).command"
    
    log_success "Desktop shortcuts created"
}

# Create application bundle (optional)
create_app_bundle() {
    log_info "Creating application bundle..."
    
    APP_BUNDLE="$APPLICATIONS_DIR/Frigate Detector.app"
    
    # Create app bundle structure
    mkdir -p "$APP_BUNDLE/Contents/MacOS"
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    
    # Create Info.plist
    cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>frigate-detector-gui</string>
    <key>CFBundleIdentifier</key>
    <string>com.apple-silicon-frigate-detector.gui</string>
    <key>CFBundleName</key>
    <string>Frigate Detector</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
EOF
    
    # Create launcher script
    cat > "$APP_BUNDLE/Contents/MacOS/frigate-detector-gui" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
exec frigate-detector-gui
EOF
    chmod +x "$APP_BUNDLE/Contents/MacOS/frigate-detector-gui"
    
    log_success "Application bundle created at $APP_BUNDLE"
}

# Create uninstaller
create_uninstaller() {
    log_info "Creating uninstaller..."
    
    cat > "$INSTALL_DIR/uninstall.sh" << EOF
#!/bin/bash
echo "Uninstalling Apple Silicon Frigate Detector..."

# Remove installation directory
rm -rf "$INSTALL_DIR"

# Remove desktop shortcuts
rm -f "$DESKTOP_DIR/Frigate Detector (CLI).command"
rm -f "$DESKTOP_DIR/Frigate Detector (GUI).command"

# Remove application bundle
rm -rf "$APPLICATIONS_DIR/Frigate Detector.app"

echo "Uninstallation complete!"
EOF
    chmod +x "$INSTALL_DIR/uninstall.sh"
    
    log_success "Uninstaller created at $INSTALL_DIR/uninstall.sh"
}

# Display post-install information
show_post_install_info() {
    echo ""
    echo "======================================"
    log_success "Installation completed successfully!"
    echo "======================================"
    echo ""
    echo "Desktop shortcuts created:"
    echo "  • Frigate Detector (GUI).command - Graphical interface"
    echo "  • Frigate Detector (CLI).command - Command line help"
    echo ""
    echo "Application bundle created:"
    echo "  • $APPLICATIONS_DIR/Frigate Detector.app"
    echo ""
    echo "Command line tools available:"
    echo "  • frigate-detector - CLI interface"
    echo "  • frigate-detector-gui - GUI interface"
    echo ""
    echo "To use from terminal:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  frigate-detector --model /path/to/your/model.onnx"
    echo ""
    echo "To uninstall:"
    echo "  $INSTALL_DIR/uninstall.sh"
    echo ""
    log_info "You can now double-click the desktop shortcuts or use the command line tools!"
}

# Main installation process
main() {
    echo "Apple Silicon Frigate Detector Installer"
    echo "========================================"
    echo ""
    
    check_requirements
    create_install_dir
    create_venv
    install_package
    create_desktop_shortcuts
    create_app_bundle
    create_uninstaller
    show_post_install_info
}

# Run main function
main "$@"
