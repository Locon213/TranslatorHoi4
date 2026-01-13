<div align="center">

# 🌍 TranslatorHoi4
### Next-Gen AI Translation Tool for Paradox Games

[![Release](https://img.shields.io/github/v/release/Locon213/TranslatorHoi4?style=for-the-badge&color=blue)](https://github.com/Locon213/TranslatorHoi4/releases/latest)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Locon213/TranslatorHoi4/build.yml?style=for-the-badge)](https://github.com/Locon213/TranslatorHoi4/actions)
[![License](https://img.shields.io/github/license/Locon213/TranslatorHoi4?style=for-the-badge&color=green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-ffd343?style=for-the-badge&logo=python&logoColor=black)](https://www.python.org/)

**[English](README.md) | [Русский](README_RU.md)**

<br>

<p align="center">
  <b>Translate huge mods in minutes, not days.</b><br>
  Stop wasting time on manual translation. Harness the power of GPT-5, Claude, and Llama 4 to localize your mod with context awareness.
</p>


<img width="1100" height="750" alt="Аннотация 2026-01-14 024843" src="https://github.com/user-attachments/assets/ecab3e08-9aad-414d-9f7f-00e35e1e24b2" />


<br>
<br>

[⬇️ Download for Windows](https://github.com/Locon213/TranslatorHoi4/releases/latest) &nbsp;&nbsp;•&nbsp;&nbsp; [🐧 Linux Instructions](#-installation) &nbsp;&nbsp;•&nbsp;&nbsp; [💬 Report Bug](https://github.com/Locon213/TranslatorHoi4/issues)

</div>

---

## ⚡ Why TranslatorHoi4?

Modding is fun. Translating thousands of lines of code is not.
Existing tools are either too simple (Google Translate breaks code) or too expensive.

**TranslatorHoi4 solves this:**
*   **Context Aware:** It knows it's translating a HOI4 mod. It tries to preserve variables, color codes (`§Y`), and formatting.
*   **Wallet Friendly:** Use **Free** providers (G4F), **Cheap** (Groq/Fireworks), or **Premium** (OpenAI/Anthropic).
*   **Modern UI:** No more command line. A beautiful Windows 11-style interface.

## 🎮 Supported Games

| Game | Status | Notes |
| :--- | :---: | :--- |
| **Hearts of Iron IV** | ✅ | Fully optimized prompts |
| **Crusader Kings III** | ⚠️ | Works, prompts behave generically |
| **Europa Universalis IV** | ⚠️ | Works, prompts behave generically |
| **Stellaris** | ⚠️ | Works, prompts behave generically |

## ✨ Key Features

### 🧠 Flexible AI Backend
*   **Cloud Power:** Support for **OpenAI** (GPT-5), **Anthropic** (Claude 4.5 Sonnet), **Google Gemini**, **DeepL**.
*   **Speed & Economy:** Blazing fast translations with **Groq** and **Fireworks.ai** (Llama 4, DeepSeek).
*   **Free & Privacy:** Use **Ollama** to run models locally on your GPU, or **G4F** for free web access.

### 🛠️ Built for Modders
*   **Smart Batching:** Handles huge localization files by splitting them into safe chunks.
*   **Glossary System:** Force specific terms (e.g., *Manpower* -> *Lidská síла*) to ensure consistency.
*   **Code Safety:** Regex filters prevent the AI from translating code keys and variables.
*   **Resume Capability:** Stopped halfway? The tool skips already translated lines next time.

## 🚀 Installation

### Windows (Recommended)
1. Go to the [**Releases Page**](https://github.com/Locon213/TranslatorHoi4/releases/latest).
2. Download the `TranslatorHoi4_Windows.zip` file.
3. Extract it and run `TranslatorHoi4.exe`.
   > *Note: If your antivirus flags the file, it is a false positive caused by PyInstaller. The code is 100% open source.*

### Linux / Source Code
```bash
# Clone the repo
git clone https://github.com/Locon213/TranslatorHoi4.git
cd TranslatorHoi4

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python -m translatorhoi4
```

## 📊 AI Providers Cheat Sheet

Not sure which provider to choose?

| Provider | Best For... | Cost | Speed |
| :--- | :--- | :--- | :--- |
| **Groq** | **The Best Value.** Extremely fast, very cheap, good quality (Llama 4). | $ | ⚡⚡⚡⚡⚡ |
| **OpenAI** | **Top Quality.** Best for complex lore and flavor text. | $$$ | ⚡⚡ |
| **Claude** | **Natural Writing.** Great for events and descriptions. | $$$ | ⚡⚡ |
| **G4F** | **Zero Budget.** Free, but unstable. Good for testing. | Free | ⚡ |
| **Ollama** | **Privacy.** Runs on your own PC. No data leaves your room. | Free | ⚡ (Depends on GPU) |

## 🤝 Contributing

We welcome pull requests! If you want to add a new AI provider or fix a bug:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes.
4. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  <b>Made with ❤️ by <a href="https://github.com/Locon213">Locon213</a></b><br>
  <i>Don't forget to star the repo if this tool saved your time! ⭐</i>
</div>
