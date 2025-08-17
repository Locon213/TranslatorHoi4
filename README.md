# TranslatorHoi4

[![Release][release-badge]][release-url]
[![Build][build-badge]][build-url]

Кроссплатформенный графический инструмент на PyQt6 для перевода файлов локализации Paradox (HOI4, CK3, EU4, Stellaris).

## For Users

1. Скачайте архив для вашей системы на [странице релизов][release-url].
2. Распакуйте его в любую папку.
3. Запустите `translatorhoi4` (Windows: `translatorhoi4.exe`).

## For Developers / Contributors

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # в Windows используйте .venv\Scripts\activate
pip install -r requirements.txt
```

### Run from sources

```bash
python -m translatorhoi4
```

### Build

```bash
pyinstaller --noconfirm translatorhoi4.spec
```

## Провайдеры и модели G4F

Пример:

```python
from g4f import Provider
provider = Provider.PollinationsAI
model = "gemini-2.5-flash"
```

Справочник: https://github.com/gpt4free/g4f.dev/blob/main/docs/providers-and-models.md

## Версия

Текущая версия: 1.0

## Лицензия

MIT

Автор: Locon213

[release-badge]: https://img.shields.io/github/v/release/Locon213/TranslatorHoi4
[release-url]: https://github.com/Locon213/TranslatorHoi4/releases
[build-badge]: https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml
