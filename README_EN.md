# TranslatorHoi4 🌍

[![Release][release-badge]][release-url]
[![Build][build-badge]][build-url]
[![Python][python-badge]][python-url]
[![License][license-badge]][license-url]

**TranslatorHoi4** is a powerful cross-platform tool with PyQt6 GUI for automatic translation of Paradox Interactive game localization files using artificial intelligence.

## 🎮 Supported Games

- **Hearts of Iron IV (HOI4)** 🎖️
- **Crusader Kings III (CK3)** 👑
- **Europa Universalis IV (EU4)** 🏰
- **Stellaris** 🚀
(Games other than HOI4 are supported but prompts are currently only written for HOI4)

## ✨ Key Features

### 🤖 Multiple AI Provider Support
- **Free**: G4F, Google Translate, Ollama
- **Fast and Affordable**: Groq, Fireworks.ai
- **Premium**: OpenAI, Anthropic Claude, Google Gemini, DeepL, Yandex Cloud
- **Local**: Ollama for complete privacy

### 🚀 Translation Modes
- **Normal mode** — translate entire mod
- **Batch mode** — translate large mods in parts
- **Chunk mode** — translate with splitting into blocks for optimization
- **Re-translation** — fix individual strings through interface

### 💡 Smart Features
- 🔍 **Automatic scanning** of localization files
- 💰 **Real-time cost tracking** of translation
- 📚 **Glossary support** for accurate terminology
- 🔄 **Intelligent caching** to speed up repeated translations
- 📝 **Post-processing** with game context in mind
- 🎨 **Multi-language interface**

### ⚙️ Advanced Settings
- Temperature and model parameter adjustment
- Key filtering through regular expressions
- Skip already translated files
- File renaming with language consideration
- Support for previous localizations

## 📋 Quick Start

### For Users

1. **Download** the ready-made build for your system on the [releases page][release-url]
2. **Extract** the archive to any folder
3. **Run** `translatorhoi4` (Windows: `translatorhoi4.exe`)

### For Developers

```bash
# Clone repository
git clone https://github.com/Locon213/TranslatorHoi4.git
cd TranslatorHoi4

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run from source
python -m translatorhoi4

# Build executable
pyinstaller --noconfirm translatorhoi4.spec
```

## 🔧 Requirements

- **Python**: 3.11 or higher
- **Operating System**: Windows, Linux
- **API Keys**: For working with paid providers

## 🎯 Usage Examples

### Basic Mod Translation
1. Select the mod folder in the "Source mod folder" field
2. Specify the original and target languages
3. Choose AI provider
4. Click "Start Translation"

### Translation with Glossary
1. Create CSV file with terms (original,translation)
2. Specify glossary path in settings
3. The program will automatically substitute your terms

### Working with Previous Localization
1. Specify folder with previous translation
2. Enable "Reuse previous localization"
3. The program will keep existing translations and add new ones

## 📊 Providers and Models

### Recommended Combinations

| Provider | Model | Speed | Quality | Cost |
|-----------|---------|----------|----------|-----------|
| **Groq** | Many Open Source models | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 |
| **Fireworks** | Also many Open Source models | ⚡⚡⚡⚡ | ⭐⭐⭐ | 💰 |
| **G4F** | Not all models work | ⚡⚡⚡ | ⭐⭐⭐⭐ | 🆓 |
| **OpenAI** | gpt-4 (And all other OpenAI models) | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 |
| **Anthropic** | claude-sonnet 4.5 | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 |

### Full Provider List
- **G4F**: Free access to various models
- **Groq**: Ultra-fast open-source models
- **Fireworks.ai**: Efficient open-source models
- **OpenAI**: GPT-5 and new models
- **Anthropic**: Claude 4.5 Sonnet, Haiku
- **Google**: Gemini 3 Pro, Flash
- **DeepL**: Professional translation
- **Yandex**: Cloud models and Translate API
- **Ollama**: Local models (Llama, Mistral, etc.)
- **Together.ai**: Access to open-source models

## 🛠️ Development and Contribution

### Project Structure
```
translatorhoi4/
├── app.py              # Main application file
├── ui/                 # Graphical interface
├── translator/         # Translation engine
├── parsers/            # Localization file parsers
├── utils/              # Utilities and helper functions
└── tests/              # Tests
```

### Adding New Provider
1. Add class in [`translatorhoi4/translator/backends.py`](translatorhoi4/translator/backends.py)
2. Register in [`MODEL_REGISTRY`](translatorhoi4/translator/engine.py)
3. Add UI elements in [`translatorhoi4/ui/ui_interfaces.py`](translatorhoi4/ui/ui_interfaces.py)

## 📚 Documentation

- [User Guide](docs/user-guide.md)
- [API Documentation](docs/api.md)
- [Provider Adding Guide](docs/adding-providers.md)

## 🤝 Project Participation

We welcome contributions to the project! Here's how you can help:

- 🐛 **Report bugs** via [Issues](https://github.com/Locon213/TranslatorHoi4/issues)
- 💡 **Suggest new features**
- 🔧 **Submit pull requests** with improvements
- 📖 **Improve documentation**
- 🌐 **Help with interface translation**

## 📄 License

The project is distributed under the **MIT** license. Details in the [LICENSE](LICENSE) file.

## 👥 Authors

- **Locon213** — Main developer
- [All contributors](https://github.com/Locon213/TranslatorHoi4/contributors)

## 🙏 Acknowledgments

- [g4f](https://github.com/xtekky/gpt4free) for free access to AI models
- [PyQt6-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets/tree/PyQt6) for beautiful interface
- Paradox Interactive community for support

---

<div align="center">
  
**⭐ If the project was helpful, give it a star on GitHub! ⭐**

[release-badge]: https://img.shields.io/github/v/release/Locon213/TranslatorHoi4
[release-url]: https://github.com/Locon213/TranslatorHoi4/releases
[build-badge]: https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml
[python-badge]: https://img.shields.io/badge/python-3.11+-blue.svg
[python-url]: https://www.python.org/downloads/
[license-badge]: https://img.shields.io/badge/license-MIT-green.svg
[license-url]: https://github.com/Locon213/TranslatorHoi4/blob/main/LICENSE

</div>
