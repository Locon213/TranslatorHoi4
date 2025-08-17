# TranslatorHoi4

Cross-platform PyQt6 GUI tool to translate Paradox localisation files (HOI4, CK3, EU4, Stellaris).

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python -m translatorhoi4
```

## G4F providers & models

Example:

```python
from g4f import Provider
provider = Provider.PollinationsAI
model = "gemini-2.5-flash"
```

Reference: https://github.com/gpt4free/g4f.dev/blob/main/docs/providers-and-models.md

## Packaging (Linux)

Ensure `assets/icon.png` exists with your custom icon.

```bash
pyinstaller --noconfirm --onedir --name TranslatorHoi4 --icon assets/icon.png translatorhoi4/app.py
```

## Screenshots

*(placeholders)*

## License

MIT

Author: Locon213
