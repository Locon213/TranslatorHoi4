#!/bin/bash
# Script to create .rpm package for TranslatorHoi4
# Usage: ./create-rpm.sh <path-to-dist> <version> <architecture>

set -e

DIST_PATH="$1"
VERSION="$2"
ARCH="$3"

if [ -z "$DIST_PATH" ] || [ -z "$VERSION" ] || [ -z "$ARCH" ]; then
    echo "Usage: $0 <dist-path> <version> <architecture>"
    echo "Example: $0 ../dist/TranslatorHoi4 1.6 x86_64"
    exit 1
fi

# Convert architecture
if [ "$ARCH" = "x64" ]; then
    RPM_ARCH="x86_64"
elif [ "$ARCH" = "arm64" ]; then
    RPM_ARCH="aarch64"
else
    RPM_ARCH="$ARCH"
fi

PACKAGE_NAME="translatorhoi4"
RPM_PACKAGE_NAME="${PACKAGE_NAME}-${VERSION}-1.${RPM_ARCH}.rpm"

echo "Building .rpm package: $RPM_PACKAGE_NAME"

# Create RPM build directory structure
RPMBUILD=$(mktemp -d)
mkdir -p "$RPMBUILD"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Create source tarball
SOURCE_NAME="${PACKAGE_NAME}-${VERSION}"
SOURCE_ROOT="$RPMBUILD/${SOURCE_NAME}"
mkdir -p "$SOURCE_ROOT"
cp -a "$DIST_PATH"/. "$SOURCE_ROOT/"
tar -czf "$RPMBUILD/SOURCES/${SOURCE_NAME}.tar.gz" -C "$RPMBUILD" "${SOURCE_NAME}"

# Create spec file
cat > "$RPMBUILD/SPECS/${PACKAGE_NAME}.spec" << EOF
Name:           ${PACKAGE_NAME}
Version:        ${VERSION}
Release:        1%{?dist}
Summary:        Cross-platform Paradox localisation translator with AI
License:        MIT
URL:            https://github.com/Locon213/TranslatorHoi4

Source0:        %{name}-%{version}.tar.gz

BuildArch:      ${RPM_ARCH}

Requires:       libglvnd-glx, libwayland-client, libwayland-cursor, libwayland-egl, libxkbcommon, libxkbcommon-x11, libXcursor, libXrandr, libXi, libXrender, libxcb, libdbus-1, pulseaudio-libs

%description
TranslatorHoi4 is a tool for translating Paradox Interactive game files
using various AI providers including OpenAI, Anthropic, Google, and others.

Supported games:
- Hearts of Iron IV (HOI4) - fully optimized
- Crusader Kings 3 (CK3)
- Europa Universalis 4 (EU4)
- Stellaris

%prep
%setup -q

%install
mkdir -p %{buildroot}/opt/translatorhoi4
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/share/applications
mkdir -p %{buildroot}/usr/share/pixmaps

cp -r %{_builddir}/${PACKAGE_NAME}-%{version}/* %{buildroot}/opt/translatorhoi4/

# Create launcher script
cat > %{buildroot}/usr/bin/translatorhoi4 << 'LAUNCHER'
#!/bin/bash
exec /opt/translatorhoi4/TranslatorHoi4 "\$@"
LAUNCHER
chmod 755 %{buildroot}/usr/bin/translatorhoi4

# Create .desktop file
cat > %{buildroot}/usr/share/applications/translatorhoi4.desktop << 'DESKTOP'
[Desktop Entry]
Name=TranslatorHoi4
Comment=Cross-platform Paradox localisation translator with AI
Exec=/opt/translatorhoi4/TranslatorHoi4
Icon=/opt/translatorhoi4/assets/icon.png
Terminal=false
Type=Application
Categories=Development;Translation;
Keywords=translation;HOI4;Paradox;AI;
DESKTOP

# Copy icon
if [ -f %{buildroot}/opt/translatorhoi4/assets/icon.png ]; then
    cp %{buildroot}/opt/translatorhoi4/assets/icon.png %{buildroot}/usr/share/pixmaps/translatorhoi4.png
fi

%files
/opt/translatorhoi4
/usr/bin/translatorhoi4
/usr/share/applications/translatorhoi4.desktop
/usr/share/pixmaps/translatorhoi4.png

%post
update-desktop-database /usr/share/applications >/dev/null 2>&1 || true

%postun
update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
EOF

# Build RPM
rpmbuild --define "_topdir $RPMBUILD" \
         --define "_builddir $RPMBUILD/BUILD" \
         --define "_target_cpu $RPM_ARCH" \
         --target "$RPM_ARCH" \
         -bb "$RPMBUILD/SPECS/${PACKAGE_NAME}.spec"

# Copy result
cp "$RPMBUILD/RPMS/${RPM_ARCH}/${RPM_PACKAGE_NAME}" "./${RPM_PACKAGE_NAME}"

echo "✓ Package created: ${RPM_PACKAGE_NAME}"
echo "  Size: $(du -h "./${RPM_PACKAGE_NAME}" | cut -f1)"

# Cleanup
rm -rf "$RPMBUILD"
