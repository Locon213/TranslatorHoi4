# Руководство по добавлению новых провайдеров

## Оглавление
1. [Введение](#введение)
2. [Подготовка](#подготовка)
3. [Создание провайдера](#создание-провайдера)
4. [Интеграция с UI](#интеграция-с-ui)
5. [Тестирование](#тестирование)
6. [Примеры](#примеры)

## Введение

Данное руководство объясняет, как добавить поддержку нового AI-провайдера в TranslatorHoi4. Процесс включает создание класса провайдера, его регистрацию и интеграцию с пользовательским интерфейсом.

## Подготовка

### Требования
- Базовые знания Python
- Понимание REST API или SDK провайдера
- Тестовый API ключ (если требуется)

### Изучение API провайдера
1. Ознакомьтесь с документацией API провайдера
2. Определите:
   - URL endpoint для перевода
   - Формат запроса и ответа
   - Необходимые заголовки
   - Модели, которые поддерживает провайдер
   - Стоимость использования (если применимо)

### Примеры провайдеров для изучения
- `G4FProvider` - простой провайдер без API ключа
- `GroqProvider` - провайдер с API ключом
- `OpenAIProvider` - провайдер с дополнительными параметрами

## Создание провайдера

### Шаг 1: Создайте класс провайдера

Создайте новый файл в `translatorhoi4/translator/backends.py` или добавьте в существующий:

```python
from typing import List, Optional, Dict, Any
import requests
from .backends import BaseProvider

class MyProvider(BaseProvider):
    """Провайдер для MyAI сервиса"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.myai.com/v1"
        self.api_key = api_key
        
    def translate(self, text: str, source_lang: str, target_lang: str, 
                  context: Optional[str] = None, **kwargs) -> str:
        """
        Выполнить перевод текста
        
        Args:
            text: Текст для перевода
            source_lang: Исходный язык
            target_lang: Целевой язык
            context: Дополнительный контекст (опционально)
            
        Returns:
            Переведенный текст
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "context": context,
            "model": kwargs.get("model", "default")
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/translate",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["translated_text"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка API MyAI: {str(e)}")
    
    def get_models(self) -> List[str]:
        """
        Получить список доступных моделей
        
        Returns:
            Список названий моделей
        """
        return ["myai-small", "myai-medium", "myai-large"]
    
    def get_cost_per_token(self) -> float:
        """
        Получить стоимость за токен (опционально)
        
        Returns:
            Стоимость в USD за 1000 токенов
        """
        return 0.001  # $0.001 за 1000 токенов
    
    def validate_credentials(self) -> bool:
        """
        Проверить валидность API ключа
        
        Returns:
            True если ключ валиден, False иначе
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.base_url}/validate",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
```

### Шаг 2: Добавьте провайдера в регистр

Откройте `translatorhoi4/translator/engine.py` и добавьте в `MODEL_REGISTRY`:

```python
from .backends import MyProvider  # Импортируйте ваш провайдер

MODEL_REGISTRY = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "g4f": G4FProvider,
    "myai": MyProvider,  # Добавьте сюда
    # ... другие провайдеры
}
```

### Шаг 3: Добавьте настройки провайдера

Если провайдеру нужны специфические настройки, добавьте их в `ProviderSettings` в `translatorhoi4/utils/settings.py`:

```python
@dataclass
class ProviderSettings:
    # Существующие настройки...
    
    # Настройки для MyAI
    myai_api_key: str = ""
    myai_api_url: str = "https://api.myai.com/v1"
    myai_temperature: float = 0.3
    myai_max_tokens: int = 1000
```

## Интеграция с UI

### Шаг 1: Добавьте UI элементы

Откройте `translatorhoi4/ui/ui_interfaces.py` и добавьте настройки для вашего провайдера:

```python
class MyProviderSettings(QWidget):
    """Настройки для MyAI провайдера"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # API ключ
        self.api_key_label = QLabel("API Key:")
        self.api_key_input = PasswordLineEdit()
        self.api_key_input.setPlaceholderText("Введите API ключ MyAI")
        
        # URL API
        self.api_url_label = QLabel("API URL:")
        self.api_url_input = LineEdit()
        self.api_url_input.setText("https://api.myai.com/v1")
        
        # Температура
        self.temp_label = QLabel("Temperature:")
        self.temp_slider = Slider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(30)
        
        # Добавляем элементы в layout
        layout.addWidget(self.api_key_label)
        layout.addWidget(self.api_key_input)
        layout.addWidget(self.api_url_label)
        layout.addWidget(self.api_url_input)
        layout.addWidget(self.temp_label)
        layout.addWidget(self.temp_slider)
        
        # Подключение сигналов
        self.api_key_input.textChanged.connect(self.on_settings_changed)
        self.api_url_input.textChanged.connect(self.on_settings_changed)
        self.temp_slider.valueChanged.connect(self.on_settings_changed)
    
    def on_settings_changed(self):
        """Сохранить настройки при изменении"""
        settings = QSettings()
        settings.setValue("myai_api_key", self.api_key_input.text())
        settings.setValue("myai_api_url", self.api_url_input.text())
        settings.setValue("myai_temperature", self.temp_slider.value() / 100)
    
    def load_settings(self):
        """Загрузить сохраненные настройки"""
        settings = QSettings()
        self.api_key_input.setText(settings.value("myai_api_key", ""))
        self.api_url_input.setText(settings.value("myai_api_url", "https://api.myai.com/v1"))
        self.temp_slider.setValue(int(settings.value("myai_temperature", 0.3) * 100))
```

### Шаг 2: Добавьте в выбор провайдера

В `translatorhoi4/ui/provider_selector.py` добавьте вашего провайдера:

```python
PROVIDER_CONFIGS = {
    "groq": {
        "name": "Groq",
        "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
        "requires_api_key": True,
        "settings_widget": GroqSettings
    },
    "myai": {  # Добавьте сюда
        "name": "MyAI",
        "models": ["myai-small", "myai-medium", "myai-large"],
        "requires_api_key": True,
        "settings_widget": MyProviderSettings,
        "cost_per_1k_tokens": 0.001
    },
    # ... другие провайдеры
}
```

### Шаг 3: Добавьте иконку провайдера

1. Создайте PNG иконку размером 32x32 или 64x64
2. Сохраните в `assets/providers/myai.png`
3. Убедитесь, что имя файла совпадает с именем провайдера

## Тестирование

### Шаг 1: Создайте тесты

Создайте файл `tests/test_my_provider.py`:

```python
import pytest
from translatorhoi4.translator.backends import MyProvider

class TestMyProvider:
    
    @pytest.fixture
    def provider(self):
        return MyProvider(api_key="test-key")
    
    def test_translate(self, provider):
        """Тест базового перевода"""
        text = "Hello world"
        result = provider.translate(text, "english", "russian")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_get_models(self, provider):
        """Тест получения списка моделей"""
        models = provider.get_models()
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_validate_credentials(self, provider):
        """Тест валидации API ключа"""
        # Это может быть интеграционный тест с реальным API
        result = provider.validate_credentials()
        assert isinstance(result, bool)
    
    def test_error_handling(self, provider):
        """Тест обработки ошибок"""
        with pytest.raises(Exception):
            provider.translate("", "english", "russian")
```

### Шаг 2: Протестируйте интеграцию

1. Запустите приложение
2. Выберите вашего провайдера из списка
3. Введите API ключ
4. Проверьте, что список моделей загружается корректно
5. Выполните тестовый перевод
6. Проверьте отображение стоимости (если применимо)

### Шаг 3: Проверьте работу с разными режимами

Протестируйте:
- Обычный режим перевода
- Пакетный режим
- Работу с глоссариями
- Кэширование переводов

## Примеры

### Пример простого провайдера без API

```python
class FreeProvider(BaseProvider):
    """Бесплатный провайдер без регистрации"""
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                  context: Optional[str] = None, **kwargs) -> str:
        # Используем библиотеку перевода
        from googletrans import Translator
        translator = Translator()
        
        result = translator.translate(
            text, 
            src=source_lang, 
            dest=target_lang
        )
        return result.text
    
    def get_models(self) -> List[str]:
        return ["google-translate-free"]
```

### Пример провайдера с дополнительными параметрами

```python
class AdvancedProvider(BaseProvider):
    """Провайдер с расширенными настройками"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        self.custom_prompt = kwargs.get("custom_prompt", None)
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                  context: Optional[str] = None, **kwargs) -> str:
        # Используем кастомный промпт если задан
        if self.custom_prompt:
            prompt = self.custom_prompt.format(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
        else:
            prompt = f"Translate from {source_lang} to {target_lang}: {text}"
        
        # Выполняем запрос с дополнительными параметрами
        data = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        # ... остальная логика
```

## Чек-лист перед публикацией

- [ ] Класс провайдера создан и наследуется от BaseProvider
- [ ] Реализованы все необходимые методы
- [ ] Провайдер добавлен в MODEL_REGISTRY
- [ ] UI настройки созданы и работают
- [ ] Иконка провайдера добавлена
- [ ] Тесты написаны и проходят
- [ ] Документация обновлена
- [ ] Проведено интеграционное тестирование

## Получение помощи

Если у вас возникли вопросы при добавлении провайдера:
- Создайте Issue на GitHub
- Изучите существующие провайдеры для примеров
- Обратитесь к сообществу в Discord