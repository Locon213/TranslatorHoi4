#!/bin/bash
# Script to create AppImage package for TranslatorHoi4
# Usage: ./create-appimage.sh <path-to-dist> <version> <architecture> [appimagetool]

set -euo pipefail

DIST_PATH="$1"
VERSION="$2"
ARCH="$3"
APPIMAGETOOL="${4:-appimagetool}"

if [ -z "$DIST_PATH" ] || [ -z "$VERSION" ] || [ -z "$ARCH" ]; then
    echo "Usage: $0 <dist-path> <version> <architecture> [appimagetool]"
    echo "Example: $0 ../dist/TranslatorHoi4 1.6 x64 ./appimagetool-x86_64.AppImage"
    exit 1
fi

if [ ! -d "$DIST_PATH" ]; then
    echo "ERROR: Dist path does not exist: $DIST_PATH"
    exit 1
fi

if [ "$ARCH" = "x64" ]; then
    APPIMAGE_ARCH="x86_64"
elif [ "$ARCH" = "arm64" ]; then
    APPIMAGE_ARCH="aarch64"
else
    APPIMAGE_ARCH="$ARCH"
fi

APP_NAME="TranslatorHoi4"
APPDIR="appimage-build/${APP_NAME}.AppDir"
APPIMAGE_NAME="${APP_NAME}-${VERSION}-${APPIMAGE_ARCH}.AppImage"

echo "Building AppImage: ${APPIMAGE_NAME}"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin/translatorhoi4"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -a "$DIST_PATH"/. "$APPDIR/usr/bin/translatorhoi4/"

cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/translatorhoi4/TranslatorHoi4" "$@"
EOF
chmod 755 "$APPDIR/AppRun"
ln -sf AppRun "$APPDIR/translatorhoi4"

cat > "$APPDIR/translatorhoi4.desktop" << 'EOF'
[Desktop Entry]
Name=TranslatorHoi4
Comment=Cross-platform Paradox localisation translator with AI
Exec=translatorhoi4
Icon=translatorhoi4
Terminal=false
Type=Application
Categories=Development;Translation;
Keywords=translation;HOI4;Paradox;AI;
EOF

cp "$DIST_PATH/assets/icon.png" "$APPDIR/translatorhoi4.png"
cp "$DIST_PATH/assets/icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/translatorhoi4.png"
cp "$APPDIR/translatorhoi4.desktop" "$APPDIR/usr/share/applications/translatorhoi4.desktop"

if [ ! -x "$APPIMAGETOOL" ]; then
    echo "ERROR: appimagetool is not executable: $APPIMAGETOOL"
    exit 1
fi

ARCH="$APPIMAGE_ARCH" APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGETOOL" "$APPDIR" "$APPIMAGE_NAME"

echo "OK Package created: ${APPIMAGE_NAME}"
echo "  Size: $(du -h "$APPIMAGE_NAME" | cut -f1)"

rm -rf "appimage-build"
