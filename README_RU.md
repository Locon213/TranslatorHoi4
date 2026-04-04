# TranslatorHoi4

[![build](https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml/badge.svg)](https://github.com/Locon213/TranslatorHoi4/actions/workflows/build.yml)
[![release](https://github.com/Locon213/TranslatorHoi4/actions/workflows/release.yml/badge.svg)](https://github.com/Locon213/TranslatorHoi4/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Кроссплатформенный переводчик локализаций Paradox (HOI4/CK3/EU4/Stellaris) с поддержкой ИИ.

## Возможности

- 🎮 **Поддерживаемые игры**: Hearts of Iron IV (полностью оптимизирован), Crusader Kings 3, Europa Universalis 4, Stellaris
- 🤖 **ИИ провайдеры**: OpenAI (GPT), Anthropic (Claude), Google Gemini, DeepL, Groq, Together.ai, Mistral AI, **NVIDIA NIM** (бесплатный и быстрый), Ollama (локальный) и другие
- 🌍 **Мультиплатформенность**: Windows (x64, arm64), Linux (x64, arm64), macOS (x64, arm64)
- ⚡ **Быстрый**: Скомпилирован через Nuitka для оптимальной производительности
- 📦 **Различные форматы**: ZIP, DMG, DEB, RPM, AppImage, установщик

## Установка

### Windows

**Вариант 1: Портативная версия (ZIP)**
1. Скачайте `TranslatorHoi4_Windows_x64.zip` (или `_arm64.zip` для ARM устройств)
2. Распакуйте в любую папку
3. Запустите `TranslatorHoi4.exe`

**Вариант 2: Установщик (Рекомендуется)**
1. Скачайте `TranslatorHoi4_Setup_<версия>.exe`
2. Запустите установщик
3. Запустите из меню Пуск или ярлыка на рабочем столе

### Linux

**Вариант 1: Портативная версия (TAR.GZ)**
```bash
tar -xzf TranslatorHoi4_Linux_x64.tar.gz
cd TranslatorHoi4
./TranslatorHoi4
```

**Вариант 2: DEB пакет (Debian/Ubuntu/Mint)**
```bash
sudo dpkg -i translatorhoi4_<версия>_amd64.deb
sudo apt-get install -f  # Исправить зависимости если нужно
translatorhoi4
```

**Вариант 3: RPM пакет (Fedora/RHEL/openSUSE)**
```bash
sudo rpm -i translatorhoi4-<версия>-1.x86_64.rpm
translatorhoi4
```

**Вариант 4: AUR (Arch Linux/Manjaro)**
```bash
# Используя yay
yay -S translatorhoi4

# Или вручную
git clone https://aur.archlinux.org/translatorhoi4.git
cd translatorhoi4
makepkg -si
```

**Необходимые зависимости** (если не используете пакеты):
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

**Установщик DMG**
1. Скачайте `TranslatorHoi4_macOS_x64.dmg` (Intel) или `_arm64.dmg` (Apple Silicon)
2. Откройте DMG файл
3. Перетащите `TranslatorHoi4.app` в папку Приложения
4. Запустите из Приложений (первый раз: правая кнопка → Открыть)

### Из исходников (Все платформы)

```bash
git clone https://github.com/Locon213/TranslatorHoi4.git
cd TranslatorHoi4
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m translatorhoi4
```

## Поддерживаемые ИИ провайдеры

| Провайдер | Модели | Бесплатно | Примечания |
|-----------|--------|-----------|------------|
| **OpenAI** | GPT-4, GPT-4o, GPT-5 | ❌ | Высокое качество, платный API |
| **Anthropic** | Claude 3.5/4 Sonnet, Opus | ❌ | Отличная работа с контекстом |
| **Google** | Gemini Pro/Flash | ✅ | Быстрый, хорош для большинства задач |
| **DeepL** | DeepL Pro | ❌ | Профессиональные переводы |
| **Groq** | Llama, Mixtral | ✅ | Экстремально быстрый вывод |
| **Together.ai** | Различные открытые модели | ✅ | Большой выбор моделей |
| **Mistral AI** | Mistral, Mixtral | ✅ | Европейский провайдер |
| **NVIDIA NIM** | Различные | ✅ **Бесплатно и Быстро** | **Рекомендуется для тестирования** |
| **Ollama** | Локальные модели | ✅ | Работает локально, без API ключа |
| **G4F** | Различные | ✅ | Бесплатный, неофициальный API |

### Настройка NVIDIA NIM

NVIDIA NIM обеспечивает **бесплатный и быстрый** вывод с простой настройкой:

1. Перейдите на [NVIDIA NIM](https://build.nvidia.com/)
2. Войдите через аккаунт NVIDIA
3. Сгенерируйте API ключ
4. В настройках TranslatorHoi4:
   - Выберите провайдер: `NVIDIA NIM`
   - Введите API ключ
   - Выберите модель (например, `meta/llama-3.1-405b-instruct`)
5. Начните переводить!

**Преимущества:**
- ✅ Полностью бесплатно (без платных тарифов)
- ✅ Быстрое время ответа
- ✅ Качественные модели (Llama 3.1, Mistral и др.)
- ✅ Официальный API, стабильная работа

## Конфигурация

1. Откройте TranslatorHoi4
2. Перейдите в Настройки (иконка шестерёнки)
3. Настройте предпочитаемый ИИ провайдер
4. Установите API ключи по необходимости
5. Настройте параметры перевода

## Сборка из исходников

Требуется Python 3.11+ и Nuitka.

```bash
# Установка зависимостей
pip install -r requirements.txt

# Сборка через Nuitka
python build.py

# Результат в dist/TranslatorHoi4/
```

## Структура проекта

```
translatorhoi4/
├── translators/     # Бэкенды перевода
├── parsers/         # Парсеры файлов Paradox
├── ui/              # GUI компоненты
├── utils/           # Утилиты и помощники
└── app.py           # Главная точка входа
```

## Лицензия

MIT License - подробности в [LICENSE](LICENSE).

## Участие в разработке

Вклад приветствуется! Пожалуйста, прочитайте документацию и отправьте PR.

---

**Сделано с ❤️ для моддеров Paradox**
