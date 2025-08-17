# TranslatorHoi4

[![Release][release-badge]][release-url]
[![Build][build-badge]][build-url]

Кроссплатформенный графический инструмент на PyQt6 для перевода файлов локализации Paradox (HOI4, CK3, EU4, Stellaris).

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # в Windows используйте .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Запуск

```bash
python -m translatorhoi4
```

## Провайдеры и модели G4F

Пример:

```python
from g4f import Provider
provider = Provider.PollinationsAI
model = "gemini-2.5-flash"
```

Справочник: https://github.com/gpt4free/g4f.dev/blob/main/docs/providers-and-models.md

## Сборка

Убедитесь, что файл `assets/icon.png` (64×64) существует.
Сборка выполняется PyInstaller в режиме `--onedir`, поэтому
результатом является каталог с необходимыми зависимостями. В GitHub
Actions для Windows создаётся архив ZIP, а для Linux и macOS – TAR.GZ.

```bash
pyinstaller --noconfirm --onedir --name TranslatorHoi4 --icon assets/icon.png translatorhoi4/app.py
```

## Версия

Текущая версия: 1.0

## Лицензия

MIT

Автор: Locon213

[release-badge]: https://img.shields.io/github/v/release/Locon213/TranslatorHoi4
[release-url]: https://github.com/Locon213/TranslatorHoi4/releases
[build-badge]: https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml

