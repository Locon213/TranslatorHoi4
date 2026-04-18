<div align="center">

# 🎮 TranslatorHoi4

**AI-переводчик локализаций для игр Paradox**

[![Статус сборки](https://img.shields.io/github/actions/workflow/status/Locon213/TranslatorHoi4/build.yml?branch=main&style=for-the-badge&logo=github&logoColor=white&label=СБОРКА&labelColor=236ad3&color=4ac41a)](https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml)
[![Релиз](https://img.shields.io/github/v/release/Locon213/TranslatorHoi4?style=for-the-badge&logo=github&logoColor=white&label=РЕЛИЗ&labelColor=181717&color=brightgreen)](https://github.com/Locon213/TranslatorHoi4/releases/latest)
[![Скачивания](https://img.shields.io/github/downloads/Locon213/TranslatorHoi4/total?style=for-the-badge&logo=github&logoColor=white&label=СКАЧИВАНИЯ&labelColor=181717&color=blue)](https://github.com/Locon213/TranslatorHoi4/releases)
[![Лицензия](https://img.shields.io/badge/ЛИЦЕНЗИЯ-MIT-brightgreen?style=for-the-badge&logo=open-source-initiative&logoColor=white&labelColor=181717)](https://github.com/Locon213/TranslatorHoi4/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/PYTHON-3.12+-blue?style=for-the-badge&logo=python&logoColor=white&labelColor=3776ab)](https://www.python.org/downloads/)

[![Платформа: Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/Locon213/TranslatorHoi4/releases)
[![Платформа: Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/Locon213/TranslatorHoi4/releases)
[![Платформа: macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Locon213/TranslatorHoi4/releases)

</div>

---

<div align="center">

![Скриншот](assets/screenshot.png)

</div>

---

## ✨ Возможности

<table>
<tr>
<td width="50%">

### 🎯 Поддержка игр
- **Hearts of Iron IV** — полностью оптимизирован
- **Crusader Kings III**
- **Europa Universalis IV**
- **Stellaris**
- **Victoria 3**
- **Imperator: Rome**
- **Пользовательские темы модов** — My Little Pony, Холодная война, Фэнтези и др.
- Работает с любой Paradox игрой на YAML

</td>
<td width="50%">

### 🤖 AI-провайдеры
- **OpenAI** (GPT-4, GPT-4o)
- **Anthropic** (Claude)
- **Google** (Gemini Pro/Flash)
- **NVIDIA NIM** — бесплатно и быстро ⭐
- **Groq, Together.ai, Mistral**
- **DeepL, Ollama (локально), G4F**

</td>
</tr>
<tr>
<td width="50%">

### 🌐 Кроссплатформенность
- **Windows** — x64, ARM64
- **Linux** — x64, ARM64, DEB, RPM, AppImage
- **macOS** — Intel, Apple Silicon
- Автоматический **Wayland/X11 fallback** на Linux
- Нативная производительность с Nuitka

</td>
<td width="50%">

### 🛠️ Для моддеров
- Современный Fluent Design UI
- **Профили перевода для каждой игры** — оптимизированные промты для Paradox
- **Поддержка темы мода** — укажите тему мода для лучшей лексики AI
- Поддержка глоссариев
- Пакетный перевод с кэшированием
- Встроенный просмотр и редактирование
- Валидация синтаксиса Paradox

</td>
</tr>
</table>

---

## 📥 Установка

### Windows

| Способ | Инструкция |
|--------|------------|
| **💾 Установщик (Рекомендуется)** | Скачай `TranslatorHoi4_Setup_*.exe` → Запусти → Открой из меню Пуск |
| **📦 Портативная** | Скачай `TranslatorHoi4_Windows_*.zip` → Распакуй → Запусти `.exe` |

### Linux

| Способ | Команда |
|--------|---------|
| **📦 DEB** (Debian/Ubuntu) | `sudo dpkg -i translatorhoi4_*.deb && sudo apt-get install -f` |
| **📦 RPM** (Fedora/openSUSE) | `sudo rpm -i translatorhoi4-*.rpm` |
| **📦 AppImage** | `chmod +x TranslatorHoi4-*.AppImage && ./TranslatorHoi4-*.AppImage` |
| **📦 Портативная** | `tar -xzf TranslatorHoi4_Linux_*.tar.gz && ./TranslatorHoi4/TranslatorHoi4` |

**Зависимости** (если не используете пакеты или запускаете распакованную portable-сборку):
```bash
# Debian/Ubuntu
sudo apt-get install libegl1 libopengl0 libgl1 libwayland-client0 \
  libwayland-cursor0 libwayland-egl1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
  libdbus-1-3 libpulse0 libxrender1 libxi6 libxrandr2

# Fedora
sudo dnf install libglvnd-glx libwayland-client libwayland-cursor \
  libwayland-egl libxkbcommon libxkbcommon-x11 libXcursor libXrandr \
  libXi libXrender libxcb dbus-libs pulseaudio-libs
```

**Примечание про Wayland/X11:** на Linux приложение теперь предпочитает Wayland внутри Wayland-сессии и автоматически откатывается на X11/XCB. В X11-сессии приоритет обратный, так что одна и та же сборка нормально работает в обеих средах.

### macOS

1. Скачай `TranslatorHoi4_macOS_*.dmg`
2. Открой DMG файл
3. Перетащи `TranslatorHoi4.app` в Приложения
4. **Первый запуск:** ПКМ → Открыть (обход Gatekeeper)

### 🧪 Из исходников

```bash
git clone https://github.com/Locon213/TranslatorHoi4.git
cd TranslatorHoi4
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m translatorhoi4
```

---

## � Улучшенные промты для AI-перевода

TranslatorHoi4 теперь использует **профили перевода для каждой игры** для более качественной локализации:

### Поддерживаемые игры

| Игра | Жанр | Ключевая терминология |
|------|------|----------------------|
| **Hearts of Iron IV** | Гранд-стратегия WW2 | Военная, политическая, историческая |
| **Crusader Kings III** | Средневековый симулятор династий | Феодальная, династическая, религиозная |
| **Europa Universalis IV** | Стратегия раннего нового времени | Колониальная, дипломатическая, торговая |
| **Stellaris** | Научно-фантастическая стратегия | Космическая, футуристическая |
| **Victoria 3** | Стратегия индустриальной эпохи | Экономическая, социальная, промышленная |
| **Imperator: Rome** | Древняя стратегия | Классическая, эллинистическая, римская |

### Пользовательские темы модов

Переводите мод **My Little Pony** для HoI4? Тотальную конверсию **Холодная война 2024**? Без проблем!

Просто укажите тему мода в поле **Mod Theme** и AI адаптирует лексику:

```
Game: Hearts of Iron IV
Mod Theme: My Little Pony, Friendship is Magic
```

AI будет использовать соответствующую лексику MLP, сохраняя игровые токены вроде `$VARS$` и `[macros]`.

### Что нового в промтах

- ✅ **8 критических правил** вместо 2 — лучшая защита от галлюцинаций AI
- ✅ **Терминология для каждой игры** — военные термины для HoI4, средневековые для CK3, научные для Stellaris
- ✅ **Сохранение токенов** — `$VARS$`, `[Scripted.Macros]`, `\n` остаются нетронутыми
- ✅ **Контекстно-зависимый перевод** — AI понимает жанр, стиль и период игры
- ✅ **Переопределение темы мода** — пользовательские темы переопределяют стандартную лексику

---

## �🤖 Рекомендуем: NVIDIA NIM

<div align="center">

| Возможность | Преимущество |
|-------------|--------------|
| 💰 **Полностью бесплатно** | Без платных тарифов, без карты |
| ⚡ **Ультра-быстро** | Оптимизированная инфраструктура |
| 🏆 **Качественные модели** | Llama 3.x, Mixtral и другие |
| 🔒 **Официальный API** | Стабильный, надёжный |

</div>

**Настройка за 60 секунд:**
1. Перейди на [build.nvidia.com](https://build.nvidia.com/)
2. Войди через аккаунт NVIDIA
3. Сгенерируй API ключ
4. Вставь ключ в настройки TranslatorHoi4 → Переводи!

---

## 🚀 Быстрый старт

1. **Запусти** TranslatorHoi4
2. **Настрой** AI-провайдер в Настройках (⚙️)
3. **Выбери** папку с модом
4. **Выбери** целевой язык
5. **Переводи!**

---

## 📊 Все поддерживаемые провайдеры

<details>
<summary>Нажми, чтобы раскрыть список</summary>

| Провайдер | Лучше всего для | Бесплатно | Скорость |
|-----------|-----------------|-----------|----------|
| **NVIDIA NIM** | Тестирование и разработка | ✅ Безлимит | ⚡⚡⚡ |
| **Groq** | Продакшен перевод | ✅ Щедро | ⚡⚡⚡ |
| **Google Gemini** | Общие задачи | ✅ Лимит | ⚡⚡ |
| **Mistral AI** | Европейские языки | ✅ Лимит | ⚡⚡ |
| **Together.ai** | Открытые модели | ✅ Лимит | ⚡⚡ |
| **OpenAI** | Премиум качество | ❌ | ⚡⚡ |
| **Anthropic** | Сложный контекст | ❌ | ⚡⚡ |
| **DeepL** | Профессиональный перевод | ❌ | ⚡⚡ |
| **Ollama** | Приватность (локально) | ✅ Безлимит | ⚡ |
| **G4F** | Альтернатива сообщества | ✅ Безлимит | ⚡ |

</details>

---

## 🏗️ Структура проекта

```
translatorhoi4/
├── translator/      # Движки перевода и бэкенды
├── parsers/         # Парсеры файлов Paradox
├── ui/              # UI компоненты Fluent Design
├── utils/           # Утилиты и помощники
└── app.py           # Точка входа приложения
```

---

## 🛠️ Сборка из исходников

```bash
pip install -r requirements.txt
python build.py      # Результат: dist/TranslatorHoi4/
```

---

## 📜 Лицензия

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

<div align="center">

**Сделано с ❤️ для сообщества моддеров Paradox**

[📥 Скачать последнюю версию](https://github.com/Locon213/TranslatorHoi4/releases/latest) • [🐛 Сообщить об ошибке](https://github.com/Locon213/TranslatorHoi4/issues) • [💬 Обсуждения](https://github.com/Locon213/TranslatorHoi4/discussions)

</div>
