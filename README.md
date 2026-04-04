# TranslatorHoi4

Cross-platform Paradox localisation translator (HOI4/CK3/EU4/Stellaris) with AI support.

## Features

- 🎮 **Supported games**: Hearts of Iron IV (fully optimized), Crusader Kings 3, Europa Universalis 4, Stellaris
- 🤖 **AI Providers**: OpenAI (GPT), Anthropic (Claude), Google Gemini, DeepL, Groq, Together.ai, Mistral AI, **NVIDIA NIM** (free & fast), Ollama (local), and more
- 🌍 **Multi-platform**: Windows (x64, arm64), Linux (x64, arm64), macOS (x64, arm64)
- ⚡ **Fast**: Compiled with Nuitka for optimal performance
- 📦 **Multiple formats**: ZIP, DMG, DEB, RPM, AppImage, Setup installer

## Installation

### Windows

**Option 1: Portable (ZIP)**
1. Download `TranslatorHoi4_Windows_x64.zip` (or `_arm64.zip` for ARM devices)
2. Extract to any folder
3. Run `TranslatorHoi4.exe`

**Option 2: Installer (Recommended)**
1. Download `TranslatorHoi4_Setup_<version>.exe`
2. Run the installer
3. Launch from Start Menu or Desktop shortcut

### Linux

**Option 1: Portable (TAR.GZ)**
```bash
tar -xzf TranslatorHoi4_Linux_x64.tar.gz
cd TranslatorHoi4
./TranslatorHoi4
```

**Option 2: DEB package (Debian/Ubuntu/Mint)**
```bash
sudo dpkg -i translatorhoi4_<version>_amd64.deb
sudo apt-get install -f  # Fix dependencies if needed
translatorhoi4
```

**Option 3: RPM package (Fedora/RHEL/openSUSE)**
```bash
sudo rpm -i translatorhoi4-<version>-1.x86_64.rpm
translatorhoi4
```

**Option 4: AUR (Arch Linux/Manjaro)**
```bash
# Using yay
yay -S translatorhoi4

# Or manually
git clone https://aur.archlinux.org/translatorhoi4.git
cd translatorhoi4
makepkg -si
```

**Required dependencies** (if not using packages):
```bash
# Debian/Ubuntu
sudo apt-get install libegl1 libopengl0 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
  libxcb-shape0 libxcb-xinerama0 libdbus-1-3 libpulse0

# Fedora
sudo dnf install libglvnd-glx libxkbcommon libXcursor libXrandr libXi libxcb libdbus-1 pulseaudio-libs

# Arch
sudo pacman -S qt6-base qt6-svg libgl libxkbcommon-x11 libxcb libpulse dbus
```

### macOS

**DMG Installer**
1. Download `TranslatorHoi4_macOS_x64.dmg` (Intel) or `_arm64.dmg` (Apple Silicon)
2. Open the DMG file
3. Drag `TranslatorHoi4.app` to Applications folder
4. Launch from Applications (first time: right-click → Open)

### From Source (All platforms)

```bash
git clone https://github.com/Locon213/TranslatorHoi4.git
cd TranslatorHoi4
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m translatorhoi4
```

## Supported AI Providers

| Provider | Models | Free Tier | Notes |
|----------|--------|-----------|-------|
| **OpenAI** | GPT-4, GPT-4o, GPT-5 | ❌ | High quality, paid API |
| **Anthropic** | Claude 3.5/4 Sonnet, Opus | ❌ | Excellent context handling |
| **Google** | Gemini Pro/Flash | ✅ | Fast, good for most tasks |
| **DeepL** | DeepL Pro | ❌ | Professional translations |
| **Groq** | Llama, Mixtral | ✅ | Extremely fast inference |
| **Together.ai** | Various open models | ✅ | Wide model selection |
| **Mistral AI** | Mistral, Mixtral | ✅ | European provider |
| **NVIDIA NIM** | Various | ✅ **Free & Fast** | **Recommended for testing** |
| **Ollama** | Local models | ✅ | Runs locally, no API key |
| **G4F** | Various | ✅ | Free, unofficial API |

### NVIDIA NIM Setup

NVIDIA NIM provides **free and fast** inference with easy setup:

1. Go to [NVIDIA NIM](https://build.nvidia.com/)
2. Sign in with NVIDIA account
3. Generate API key
4. In TranslatorHoi4 settings:
   - Select provider: `NVIDIA NIM`
   - Enter API key
   - Choose model (e.g., `meta/llama-3.1-405b-instruct`)
5. Start translating!

**Advantages:**
- ✅ Completely free (no paid tiers)
- ✅ Fast response times
- ✅ High-quality models (Llama 3.1, Mistral, etc.)
- ✅ Official API, stable

## Configuration

1. Open TranslatorHoi4
2. Go to Settings (gear icon)
3. Configure your preferred AI provider
4. Set API keys as needed
5. Adjust translation parameters

## Building from Source

Requires Python 3.11+ and Nuitka.

```bash
# Install dependencies
pip install -r requirements.txt

# Build with Nuitka
python build.py

# Output in dist/TranslatorHoi4/
```

## Project Structure

```
translatorhoi4/
├── translators/     # Translation backends
├── parsers/         # Paradox file parsers
├── ui/              # GUI components
├── utils/           # Utilities and helpers
└── app.py           # Main entry point
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the docs and submit PRs.

---

**Made with ❤️ for Paradox modders**
