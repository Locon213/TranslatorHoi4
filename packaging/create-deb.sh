#!/bin/bash
# Script to create .deb package for TranslatorHoi4
# Usage: ./create-deb.sh <path-to-dist> <version> <architecture>

set -e

DIST_PATH="$1"
VERSION="$2"
ARCH="$3"

normalize_package_version() {
    local raw="$1"
    local cleaned

    cleaned="$(printf '%s' "$raw" | sed -E 's/[^A-Za-z0-9.+~]+/./g; s/^[.]+//; s/[.]+$//; s/[.]{2,}/./g')"
    if [[ ! "$cleaned" =~ ^[0-9] ]]; then
        cleaned="0.0.0.${cleaned:-dev}"
    fi

    printf '%s\n' "$cleaned"
}

if [ -z "$DIST_PATH" ] || [ -z "$VERSION" ] || [ -z "$ARCH" ]; then
    echo "Usage: $0 <dist-path> <version> <architecture>"
    echo "Example: $0 ../dist/TranslatorHoi4 1.6 amd64"
    exit 1
fi

# Convert architecture
if [ "$ARCH" = "x64" ]; then
    DEB_ARCH="amd64"
elif [ "$ARCH" = "arm64" ]; then
    DEB_ARCH="arm64"
else
    DEB_ARCH="$ARCH"
fi

PACKAGE_NAME="translatorhoi4"
PACKAGE_VERSION="$(normalize_package_version "$VERSION")"
DEB_DIR="deb-build"
DEB_PACKAGE_NAME="${PACKAGE_NAME}_${PACKAGE_VERSION}_${DEB_ARCH}.deb"

echo "Building .deb package: $DEB_PACKAGE_NAME"
if [ "$PACKAGE_VERSION" != "$VERSION" ]; then
    echo "Normalized package version: $VERSION -> $PACKAGE_VERSION"
fi

# Clean previous build
rm -rf "$DEB_DIR"
mkdir -p "$DEB_DIR/DEBIAN"
mkdir -p "$DEB_DIR/usr/bin"
mkdir -p "$DEB_DIR/usr/share/applications"
mkdir -p "$DEB_DIR/usr/share/pixmaps"
mkdir -p "$DEB_DIR/opt/translatorhoi4"

# Copy application files
cp -r "$DIST_PATH"/* "$DEB_DIR/opt/translatorhoi4/"

# Create launcher script
cat > "$DEB_DIR/usr/bin/translatorhoi4" << 'EOF'
#!/bin/bash
exec /opt/translatorhoi4/TranslatorHoi4 "$@"
EOF
chmod 755 "$DEB_DIR/usr/bin/translatorhoi4"

# Create .desktop file
cat > "$DEB_DIR/usr/share/applications/translatorhoi4.desktop" << EOF
[Desktop Entry]
Name=TranslatorHoi4
Comment=Cross-platform Paradox localisation translator with AI
Exec=/opt/translatorhoi4/TranslatorHoi4
Icon=/opt/translatorhoi4/assets/icon.png
Terminal=false
Type=Application
Categories=Development;Translation;
Keywords=translation;HOI4;Paradox;AI;
EOF

# Copy icon
cp "$DIST_PATH/assets/icon.png" "$DEB_DIR/usr/share/pixmaps/translatorhoi4.png" 2>/dev/null || true

# Create control file
cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: translatorhoi4
Version: $PACKAGE_VERSION
Section: devel
Priority: optional
Architecture: $DEB_ARCH
Depends: libegl1, libopengl0, libgl1, libwayland-client0, libwayland-cursor0, libwayland-egl1, libxkbcommon0, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-render-util0, libxcb-shape0, libxcb-xinerama0, libdbus-1-3, libpulse0, libxrender1, libxi6, libxrandr2, libc6 (>= 2.17)
Maintainer: Locon213
Description: Cross-platform Paradox localisation translator (HOI4/CK3/EU4/Stellaris) with AI
 TranslatorHoi4 is a tool for translating Paradox Interactive game files
 using various AI providers including OpenAI, Anthropic, Google, and others.
 .
 Supported games:
  - Hearts of Iron IV (HOI4) - fully optimized
  - Crusader Kings 3 (CK3)
  - Europa Universalis 4 (EU4)
  - Stellaris
EOF

# Create postinst script
cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e
update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
echo "TranslatorHoi4 installed successfully!"
echo "You can launch it from the application menu or by running 'translatorhoi4' in terminal."
EOF
chmod 755 "$DEB_DIR/DEBIAN/postinst"

# Create postrm script
cat > "$DEB_DIR/DEBIAN/postrm" << 'EOF'
#!/bin/bash
set -e
if [ "$1" = "remove" ]; then
    update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
fi
EOF
chmod 755 "$DEB_DIR/DEBIAN/postrm"

# Build the package
dpkg-deb --build "$DEB_DIR" "$DEB_PACKAGE_NAME"

echo "✓ Package created: $DEB_PACKAGE_NAME"
echo "  Size: $(du -h "$DEB_PACKAGE_NAME" | cut -f1)"

# Cleanup
rm -rf "$DEB_DIR"
