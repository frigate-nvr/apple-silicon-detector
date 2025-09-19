#!/bin/bash

# One-line installer for Apple Silicon Frigate Detector
# Usage: curl -sSL https://github.com/frigate-nvr/apple-silicon-frigate-detector/releases/latest/download/install-online.sh | bash

set -e

# Configuration
REPO="frigate-nvr/apple-silicon-frigate-detector"
GITHUB_API="https://api.github.com/repos/$REPO"
TEMP_DIR=$(mktemp -d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🚀 Apple Silicon Frigate Detector Installer"
echo "============================================"

log_info "Downloading installer from GitHub releases..."

# Get latest release info
LATEST_RELEASE=$(curl -s "$GITHUB_API/releases/latest" | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4)

if [ -z "$LATEST_RELEASE" ]; then
    log_error "Failed to get latest release information"
    exit 1
fi

log_info "Latest version: $LATEST_RELEASE"

# Download the installer script
INSTALLER_URL="https://github.com/$REPO/releases/download/$LATEST_RELEASE/install.sh"

cd "$TEMP_DIR"
curl -sSL -o install.sh "$INSTALLER_URL" || {
    log_error "Failed to download installer from $INSTALLER_URL"
    log_info "Trying fallback method..."
    
    # Fallback: download from main branch
    curl -sSL -o install.sh "https://raw.githubusercontent.com/$REPO/main/install.sh" || {
        log_error "Failed to download installer. Please check your internet connection."
        exit 1
    }
}

# Make executable and run
chmod +x install.sh
./install.sh

# Cleanup
cd /
rm -rf "$TEMP_DIR"

log_success "Installation complete!"
echo ""
echo "🎉 You can now use:"
echo "   • Double-click desktop shortcuts"
echo "   • Open 'Frigate Detector.app' from Applications"
echo "   • Use 'frigate-detector' command in terminal"
