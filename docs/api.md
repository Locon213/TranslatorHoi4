# API Документация TranslatorHoi4

## Оглавление
1. [Введение](#введение)
2. [Архитектура](#архитектура)
3. [Основные классы](#основные-классы)
4. [Провайдеры перевода](#провайдеры-перевода)
5. [Парсеры](#парсеры)
6. [Кэширование](#кэширование)
7. [Утилиты](#утилиты)

## Введение

TranslatorHoi4 построен на модульной архитектуре, которая позволяет легко расширять функциональность и добавлять новые провайдеры перевода.

## Архитектура

### Общая структура
```
translatorhoi4/
├── app.py                 # Точка входа в приложение
├── translator/            # Движок перевода
│   ├── engine.py         # Основной движок
│   ├── backends.py       # Провайдеры AI
│   ├── cache.py          # Кэширование
│   └── cost.py           # Отслеживание стоимости
├── ui/                    # Пользовательский интерфейс
├── parsers/               # Парсеры файлов
└── utils/                 # Утилиты
```

### Поток данных
1. Пользователь выбирает файлы для перевода
2. Парсер извлекает текст из файлов локализации
3. Движок перевода обрабатывает текст партиями
4. Провайдер AI выполняет перевод
5. Результаты сохраняются в кэш и выходные файлы

## Основные классы

### TranslationEngine
Главный класс для управления процессом перевода.

```python
from translatorhoi4.translator.engine import TranslationEngine

engine = TranslationEngine(
    provider="groq",
    model="llama-3.1-70b-versatile",
    source_lang="english",
    target_lang="russian"
)

# Выполнить перевод
results = engine.translate_files(
    input_files=["file1.yml", "file2.yml"],
    output_dir="translated/"
)
```

### TranslationConfig
Конфигурация для перевода.

```python
from translatorhoi4.translator.engine import TranslationConfig

config = TranslationConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.3,
    chunk_size=10,
    use_cache=True,
    use_glossary=True,
    glossary_path="glossary.csv"
)
```

## Провайдеры перевода

### Базовый класс BaseProvider
Все провайдеры наследуются от `BaseProvider`:

```python
from translatorhoi4.translator.backends import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self, api_key=None, **kwargs):
        super().__init__(api_key, **kwargs)
    
    def translate(self, text, source_lang, target_lang, context=None):
        # Реализация перевода
        return translated_text
    
    def get_models(self):
        # Возвращает список доступных моделей
        return ["model1", "model2"]
```

### Доступные провайдеры

#### GroqProvider
```python
from translatorhoi4.translator.backends import GroqProvider

provider = GroqProvider(api_key="your-api-key")
translation = provider.translate(
    text="Hello world",
    source_lang="english",
    target_lang="russian"
)
```

#### OpenAIProvider
```python
from translatorhoi4.translator.backends import OpenAIProvider

provider = OpenAIProvider(api_key="your-api-key")
translation = provider.translate(
    text="Hello world",
    source_lang="english",
    target_lang="russian",
    model="gpt-4"
)
```

#### G4FProvider
```python
from translatorhoi4.translator.backends import G4FProvider

provider = G4FProvider()  # Не требует API ключа
translation = provider.translate(
    text="Hello world",
    source_lang="english",
    target_lang="russian",
    model="gpt-3.5-turbo"
)
```

### Регистрация нового провайдера
1. Создайте класс провайдера в `backends.py`
2. Добавьте в `MODEL_REGISTRY` в `engine.py`:

```python
MODEL_REGISTRY = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "g4f": G4FProvider,
    "my_provider": MyProvider,  # Ваш новый провайдер
}
```

## Парсеры

### ParadoxParser
Основной парсер для файлов локализации Paradox:

```python
from translatorhoi4.parsers.paradox_yaml import ParadoxParser

parser = ParadoxParser()
data = parser.parse("localisation/english/events_l_english.yml")
```

### Поддерживаемые форматы
- `.yml` - YAML файлы локализации
- `.yaml` - YAML файлы
- `.txt` - Текстовые файлы с особым форматированием

### Пример структуры данных
```python
{
    "KEY1": "Текст на английском",
    "KEY2": "Еще один текст",
    "KEY3": {
        "0": "Вариант 0",
        "1": "Вариант 1"
    }
}
```

## Кэширование

### SQLiteCache
Кэш на основе SQLite для постоянного хранения:

```python
from translatorhoi4.translator.sqlite_cache import SQLiteCache

cache = SQLiteCache("translations_cache.db")
cache.set("hello_world_ru", "Привет мир")
translation = cache.get("hello_world_ru")
```

### MemoryCache
Кэш в памяти для быстрого доступа:

```python
from translatorhoi4.translator.cache import MemoryCache

cache = MemoryCache(max_size=1000)
cache.set("key", "value")
value = cache.get("key")
```

### Использование кэша в переводе
```python
engine = TranslationEngine(
    provider="groq",
    use_cache=True,
    cache_type="sqlite"
)
```

## Утилиты

### CostTracker
Отслеживание стоимости перевода:

```python
from translatorhoi4.translator.cost import CostTracker

tracker = CostTracker()
tracker.add_request(provider="openai", model="gpt-4", tokens=1000)
cost = tracker.get_total_cost()
```

### Validation
Валидация настроек и данных:

```python
from translatorhoi4.utils.validation import validate_language_code

is_valid = validate_language_code("russian")  # True
is_valid = validate_language_code("invalid")  # False
```

### FileSystem
Работа с файловой системой:

```python
from translatorhoi4.utils.fs import scan_localization_files

files = scan_localization_files("mod_folder/", extension=".yml")
```

## Примеры использования

### Базовый перевод
```python
from translatorhoi4.translator.engine import TranslationEngine

engine = TranslationEngine(
    provider="groq",
    model="llama-3.1-70b-versatile",
    source_lang="english",
    target_lang="russian"
)

# Перевод текста
result = engine.translate_text("Hello world")

# Перевод файла
engine.translate_file("input.yml", "output.yml")
```

### Перевод с глоссарием
```python
from translatorhoi4.translator.glossary import Glossary

glossary = Glossary("my_glossary.csv")
engine = TranslationEngine(
    provider="openai",
    use_glossary=True,
    glossary=glossary
)
```

### Пакетный перевод
```python
engine = TranslationEngine(
    provider="groq",
    batch_size=10,
)

results = engine.translate_batch([
    "Text 1",
    "Text 2",
    "Text 3"
])
```

## Расширение функциональности

### Добавление нового парсера
1. Создайте класс в `parsers/`
2. Реализуйте методы `parse()` и `serialize()`
3. Зарегистрируйте в `PARSER_REGISTRY`

### Добавление нового провайдера
1. Наследуйтесь от `BaseProvider`
2. Реализуйте методы `translate()` и `get_models()`
3. Добавьте в `MODEL_REGISTRY`

### Создание кастомного кэша
1. Реализуйте интерфейс с методами `get()`, `set()`, `delete()`
2. Используйте в `TranslationEngine`

## Лицензия

API TranslatorHoi4 распространяется под лицензией MIT. Подробности в файле LICENSE.