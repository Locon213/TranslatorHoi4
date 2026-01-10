"""Input validation and path validation utilities."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class PathValidator:
    """Validate file and directory paths."""

    # Characters not allowed in file/directory names
    INVALID_CHARS = set('<>|?*')
    # Note: ':' is allowed in paths on Windows, '"' is invalid
    
    # Reserved names on Windows
    RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    
    @classmethod
    def validate_directory(
        cls,
        path: Union[str, Path],
        must_exist: bool = True,
        create_if_missing: bool = False,
        allow_empty: bool = False,
    ) -> Path:
        """Validate and return a directory path.
        
        Args:
            path: Path to validate
            must_exist: If True, directory must exist
            create_if_missing: If True, create directory if it doesn't exist
            allow_empty: If False, directory must not be empty
            
        Returns:
            Path object for the validated directory
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(path, str):
            path = Path(path)
        
        # Normalize path
        path = path.resolve()
        
        # Check for empty path
        if not str(path):
            raise ValidationError("Path cannot be empty")
        
        # Check for invalid characters
        path_str = str(path)
        for char in cls.INVALID_CHARS:
            if char in path_str:
                raise ValidationError(f"Path contains invalid character: {char}")
        
        if must_exist or create_if_missing:
            if not path.exists():
                if create_if_missing:
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except PermissionError as e:
                        raise ValidationError(f"Permission denied creating directory: {e}")
                    except Exception as e:
                        raise ValidationError(f"Failed to create directory: {e}")
                else:
                    raise ValidationError(f"Directory does not exist: {path}")
            
            if not path.is_dir():
                raise ValidationError(f"Path is not a directory: {path}")
            
            if not allow_empty:
                try:
                    contents = list(path.iterdir())
                    if not contents:
                        raise ValidationError(f"Directory is empty: {path}")
                except PermissionError:
                    raise ValidationError(f"Permission denied reading directory: {path}")
        
        return path
    
    @classmethod
    def validate_file(
        cls,
        path: Union[str, Path],
        must_exist: bool = True,
        allowed_extensions: Optional[set] = None,
        min_size: int = 0,
        max_size: Optional[int] = None,
    ) -> Path:
        """Validate and return a file path.
        
        Args:
            path: Path to validate
            must_exist: If True, file must exist
            allowed_extensions: Set of allowed file extensions (e.g., {'.yml', '.yaml'})
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            
        Returns:
            Path object for the validated file
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(path, str):
            path = Path(path)
        
        # Normalize path
        path = path.resolve()
        
        # Check for empty path
        if not str(path):
            raise ValidationError("File path cannot be empty")
        
        # Check for invalid characters
        path_str = str(path)
        for char in cls.INVALID_CHARS:
            if char in path_str:
                raise ValidationError(f"Path contains invalid character: {char}")
        
        # Check file extension
        if allowed_extensions:
            ext = path.suffix.lower()
            if ext not in allowed_extensions:
                raise ValidationError(
                    f"Invalid file extension '{ext}'. Allowed: {', '.join(allowed_extensions)}"
                )
        
        if must_exist:
            if not path.exists():
                raise ValidationError(f"File does not exist: {path}")
            
            if not path.is_file():
                raise ValidationError(f"Path is not a file: {path}")
            
            # Check file size
            try:
                size = path.stat().st_size
                if size < min_size:
                    raise ValidationError(f"File too small (minimum {min_size} bytes)")
                if max_size is not None and size > max_size:
                    raise ValidationError(f"File too large (maximum {max_size} bytes)")
            except PermissionError:
                raise ValidationError(f"Permission denied accessing file: {path}")
        
        # Check reserved names
        if path.stem.upper() in cls.RESERVED_NAMES:
            raise ValidationError(f"Reserved file name: {path.stem}")
        
        return path
    
    @classmethod
    def validate_output_path(
        cls,
        path: Union[str, Path],
        allowed_extensions: Optional[set] = None,
    ) -> Tuple[Path, bool]:
        """Validate output path and check if it's writable.
        
        Args:
            path: Output path to validate
            allowed_extensions: Allowed file extensions
            
        Returns:
            Tuple of (validated Path, was_created) indicating if path was created
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(path, str):
            path = Path(path)
        
        path = path.resolve()
        was_created = False
        
        # Check if parent directory exists
        parent = path.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
                was_created = True
            except PermissionError as e:
                raise ValidationError(f"Permission denied creating directory: {e}")
            except Exception as e:
                raise ValidationError(f"Failed to create directory: {e}")
        
        # If path exists, check if it's a file or directory
        if path.exists():
            if path.is_dir():
                # It's a directory, can write files here
                return path, False
            elif path.is_file():
                # It's a file, check if writable
                if not os.access(path, os.W_OK):
                    raise ValidationError(f"File is not writable: {path}")
                return path, False
        
        # For file paths, check if parent is writable
        if path.suffix:  # It's a file path
            if not os.access(parent, os.W_OK):
                raise ValidationError(f"Parent directory is not writable: {parent}")
        
        return path, was_created


class InputValidator:
    """Validate user input values."""
    
    # Valid language codes
    VALID_LANGUAGES = {
        "english", "russian", "german", "french", "spanish", "italian",
        "polish", "brazilian", "portuguese", "japanese", "chinese",
        "korean", "arabic", "turkish", "dutch", "swedish", "norwegian",
        "finnish", "danish", "czech", "hungarian", "romanian",
    }
    
    # Valid model names
    VALID_MODELS = {
        "G4F: API (g4f.dev)",
        "IO: chat.completions",
        "OpenAI Compatible API",
        "Anthropic: Claude",
        "Google: Gemini",
    }
    
    @classmethod
    def validate_language(cls, lang: str, field_name: str = "Language") -> str:
        """Validate language code.
        
        Args:
            lang: Language code to validate
            field_name: Name of the field for error messages
            
        Returns:
            Normalized language code
            
        Raises:
            ValidationError: If validation fails
        """
        if not lang:
            raise ValidationError(f"{field_name} cannot be empty")
        
        lang_lower = lang.lower().strip()
        
        if lang_lower not in cls.VALID_LANGUAGES:
            # Try to find a close match
            raise ValidationError(
                f"Invalid {field_name.lower()}: '{lang}'. "
                f"Supported: {', '.join(sorted(cls.VALID_LANGUAGES))}"
            )
        
        return lang_lower
    
    @classmethod
    def validate_model(cls, model: str) -> str:
        """Validate model name.
        
        Args:
            model: Model name to validate
            
        Returns:
            Validated model name
            
        Raises:
            ValidationError: If validation fails
        """
        if not model:
            raise ValidationError("Model cannot be empty")
        
        model_normalized = model.strip()
        
        if model_normalized not in cls.VALID_MODELS:
            raise ValidationError(
                f"Invalid model: '{model}'. "
                f"Supported: {', '.join(cls.VALID_MODELS)}"
            )
        
        return model_normalized
    
    @classmethod
    def validate_api_key(cls, api_key: str, provider: str) -> str:
        """Validate API key format.
        
        Args:
            api_key: API key to validate
            provider: Provider name for context
            
        Returns:
            Normalized API key
            
        Raises:
            ValidationError: If validation fails
        """
        if not api_key:
            return None  # API key can be optional for some providers
        
        api_key = api_key.strip()
        
        # Basic length check
        if len(api_key) < 8:
            raise ValidationError(
                f"Invalid API key for {provider}: too short (minimum 8 characters)"
            )
        
        # Check for obviously invalid formats
        if " " in api_key:
            raise ValidationError(
                f"Invalid API key for {provider}: contains spaces"
            )
        
        return api_key
    
    @classmethod
    def validate_regex(cls, pattern: str, field_name: str = "Regex") -> Optional[str]:
        """Validate regex pattern.
        
        Args:
            pattern: Regex pattern to validate
            field_name: Name of the field for error messages
            
        Returns:
            Compiled regex pattern or None if empty
            
        Raises:
            ValidationError: If validation fails
        """
        if not pattern:
            return None
        
        pattern = pattern.strip()
        
        if not pattern:
            return None
        
        try:
            re.compile(pattern)
            return pattern
        except re.error as e:
            raise ValidationError(f"Invalid {field_name.lower()}: {e}")
    
    @classmethod
    def validate_temperature(cls, temp: float, min_temp: float = 0.0, max_temp: float = 1.0) -> float:
        """Validate temperature value.
        
        Args:
            temp: Temperature value
            min_temp: Minimum allowed temperature
            max_temp: Maximum allowed temperature
            
        Returns:
            Validated temperature
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(temp, (int, float)):
            raise ValidationError("Temperature must be a number")
        
        if temp < min_temp:
            raise ValidationError(f"Temperature too low (minimum {min_temp})")
        
        if temp > max_temp:
            raise ValidationError(f"Temperature too high (maximum {max_temp})")
        
        return float(temp)
    
    @classmethod
    def validate_concurrency(cls, value: int, min_value: int = 1, max_value: int = 100) -> int:
        """Validate concurrency value.
        
        Args:
            value: Concurrency value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated concurrency value
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int):
            raise ValidationError("Concurrency must be an integer")
        
        if value < min_value:
            raise ValidationError(f"Concurrency too low (minimum {min_value})")
        
        if value > max_value:
            raise ValidationError(f"Concurrency too high (maximum {max_value})")
        
        return value
    
    @classmethod
    def validate_batch_size(cls, value: int, min_value: int = 1, max_value: int = 500) -> int:
        """Validate batch size value.
        
        Args:
            value: Batch size value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated batch size
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int):
            raise ValidationError("Batch size must be an integer")
        
        if value < min_value:
            raise ValidationError(f"Batch size too low (minimum {min_value})")
        
        if value > max_value:
            raise ValidationError(f"Batch size too high (maximum {max_value})")
        
        return value


def validate_settings(settings: dict) -> dict:
    """Validate all settings in a dictionary.
    
    Args:
        settings: Dictionary of settings to validate
        
    Returns:
        Validated settings dictionary
        
    Raises:
        ValidationError: If any setting is invalid
    """
    validated = {}
    
    # Validate source directory
    if "src" in settings:
        try:
            validated["src"] = PathValidator.validate_directory(
                settings["src"], must_exist=True, allow_empty=True
            )
        except ValidationError as e:
            raise ValidationError(f"Invalid source directory: {e}")
    
    # Validate output directory
    if "out" in settings:
        try:
            out_path = settings.get("out") or settings.get("src")
            validated["out"] = PathValidator.validate_directory(
                out_path, must_exist=False, create_if_missing=True
            )
        except ValidationError as e:
            raise ValidationError(f"Invalid output directory: {e}")
    
    # Validate languages
    if "src_lang" in settings:
        validated["src_lang"] = InputValidator.validate_language(
            settings["src_lang"], "Source language"
        )
    
    if "dst_lang" in settings:
        validated["dst_lang"] = InputValidator.validate_language(
            settings["dst_lang"], "Target language"
        )
    
    # Validate model
    if "model" in settings:
        validated["model"] = InputValidator.validate_model(settings["model"])
    
    # Validate temperature
    if "temp_x100" in settings:
        validated["temp_x100"] = InputValidator.validate_temperature(
            settings["temp_x100"] / 100.0
        ) * 100
    
    # Validate concurrency settings
    for field in ["batch_size", "files_cc", "g4f_cc", "io_cc", "openai_cc", "anthropic_cc", "gemini_cc"]:
        if field in settings:
            try:
                validated[field] = InputValidator.validate_concurrency(settings[field])
            except ValidationError as e:
                raise ValidationError(f"Invalid {field}: {e}")
    
    # Validate API keys
    for provider in ["g4f", "openai", "anthropic", "gemini", "io"]:
        key_field = f"{provider}_api_key"
        if key_field in settings:
            try:
                validated[key_field] = InputValidator.validate_api_key(
                    settings[key_field], provider
                )
            except ValidationError as e:
                raise ValidationError(f"Invalid {provider} API key: {e}")
    
    # Validate regex patterns
    if "key_skip_regex" in settings:
        validated["key_skip_regex"] = InputValidator.validate_regex(
            settings["key_skip_regex"], "Skip regex"
        )
    
    return validated
