"""Tests for validation.py - Input and path validation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestPathValidator:
    """Test path validation functionality."""
    
    def test_validate_directory_exists(self, tmp_path):
        """Test validating an existing directory."""
        from translatorhoi4.utils.validation import PathValidator
        
        result = PathValidator.validate_directory(tmp_path, must_exist=True)
        assert result == tmp_path
    
    def test_validate_directory_not_exists_with_create(self, tmp_path):
        """Test creating directory when it doesn't exist."""
        from translatorhoi4.utils.validation import PathValidator
        
        new_dir = tmp_path / "new_dir" / "nested"
        result = PathValidator.validate_directory(new_dir, must_exist=False, create_if_missing=True)
        assert result.exists()
        assert result.is_dir()
    
    def test_validate_directory_not_exists_no_create(self, tmp_path):
        """Test failing when directory doesn't exist and can't be created."""
        from translatorhoi4.utils.validation import PathValidator, ValidationError
        
        new_dir = tmp_path / "nonexistent"
        with pytest.raises(ValidationError):
            PathValidator.validate_directory(new_dir, must_exist=True)
    
    def test_validate_file_exists(self, tmp_path):
        """Test validating an existing file."""
        from translatorhoi4.utils.validation import PathValidator
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = PathValidator.validate_file(test_file, must_exist=True)
        assert result == test_file
    
    def test_validate_file_not_exists(self, tmp_path):
        """Test failing when file doesn't exist."""
        from translatorhoi4.utils.validation import PathValidator, ValidationError
        
        missing_file = tmp_path / "missing.txt"
        with pytest.raises(ValidationError):
            PathValidator.validate_file(missing_file, must_exist=True)
    
    def test_validate_file_with_extensions(self, tmp_path):
        """Test validating file with allowed extensions."""
        from translatorhoi4.utils.validation import PathValidator
        
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text("key: value")
        
        # Should pass with correct extension
        result = PathValidator.validate_file(
            yaml_file, must_exist=True, allowed_extensions={'.yml', '.yaml'}
        )
        assert result == yaml_file
        
        # Should fail with incorrect extension
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")
        with pytest.raises(ValidationError):
            PathValidator.validate_file(
                txt_file, must_exist=True, allowed_extensions={'.yml', '.yaml'}
            )
    
    def test_validate_file_size(self, tmp_path):
        """Test validating file size constraints."""
        from translatorhoi4.utils.validation import PathValidator, ValidationError
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("x" * 100)
        
        # Should pass within range
        result = PathValidator.validate_file(
            test_file, must_exist=True, min_size=50, max_size=200
        )
        assert result == test_file
        
        # Should fail if too small
        with pytest.raises(ValidationError):
            PathValidator.validate_file(
                test_file, must_exist=True, min_size=200
            )
        
        # Should fail if too large
        with pytest.raises(ValidationError):
            PathValidator.validate_file(
                test_file, must_exist=True, max_size=50
            )
    
    def test_validate_path_with_invalid_chars(self, tmp_path):
        """Test rejecting paths with invalid characters."""
        from translatorhoi4.utils.validation import PathValidator, ValidationError
        
        # Test Windows-style invalid characters
        invalid_paths = [
            tmp_path / "test<file>.txt",
            tmp_path / "test>file.txt",
            tmp_path / "test:file.txt",
            tmp_path / 'test"file.txt',
            tmp_path / "test|file.txt",
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises(ValidationError):
                PathValidator.validate_file(invalid_path)


class TestInputValidator:
    """Test input validation functionality."""
    
    def test_validate_language_valid(self):
        """Test validating a valid language."""
        from translatorhoi4.utils.validation import InputValidator
        
        assert InputValidator.validate_language("english") == "english"
        assert InputValidator.validate_language("Russian") == "russian"
        assert InputValidator.validate_language("  german  ") == "german"
    
    def test_validate_language_invalid(self):
        """Test rejecting an invalid language."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_language("invalid_language")
    
    def test_validate_language_empty(self):
        """Test rejecting empty language."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_language("")
    
    def test_validate_model_valid(self):
        """Test validating a valid model."""
        from translatorhoi4.utils.validation import InputValidator
        
        assert InputValidator.validate_model("G4F: API (g4f.dev)") == "G4F: API (g4f.dev)"
        assert InputValidator.validate_model("OpenAI Compatible API") == "OpenAI Compatible API"
    
    def test_validate_model_invalid(self):
        """Test rejecting an invalid model."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_model("Invalid Model")
    
    def test_validate_api_key_valid(self):
        """Test validating a valid API key."""
        from translatorhoi4.utils.validation import InputValidator
        
        result = InputValidator.validate_api_key("sk-test123456789", "test_provider")
        assert result == "sk-test123456789"
    
    def test_validate_api_key_empty(self):
        """Test accepting empty API key as optional."""
        from translatorhoi4.utils.validation import InputValidator
        
        result = InputValidator.validate_api_key("", "test_provider")
        assert result is None
    
    def test_validate_api_key_too_short(self):
        """Test rejecting too short API key."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_api_key("short", "test_provider")
    
    def test_validate_api_key_with_spaces(self):
        """Test rejecting API key with spaces."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_api_key("key with spaces", "test_provider")
    
    def test_validate_regex_valid(self):
        """Test validating a valid regex pattern."""
        from translatorhoi4.utils.validation import InputValidator
        
        result = InputValidator.validate_regex("^STATE_\\d+")
        assert result == "^STATE_\\d+"
    
    def test_validate_regex_empty(self):
        """Test accepting empty regex as None."""
        from translatorhoi4.utils.validation import InputValidator
        
        result = InputValidator.validate_regex("")
        assert result is None
    
    def test_validate_regex_invalid(self):
        """Test rejecting invalid regex pattern."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_regex("[unclosed")
    
    def test_validate_temperature_valid(self):
        """Test validating a valid temperature."""
        from translatorhoi4.utils.validation import InputValidator
        
        assert InputValidator.validate_temperature(0.7) == 0.7
        assert InputValidator.validate_temperature(0) == 0.0
        assert InputValidator.validate_temperature(1.0) == 1.0
    
    def test_validate_temperature_out_of_range(self):
        """Test rejecting temperature out of range."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_temperature(-0.1)
        
        with pytest.raises(ValidationError):
            InputValidator.validate_temperature(1.5)
    
    def test_validate_concurrency_valid(self):
        """Test validating a valid concurrency value."""
        from translatorhoi4.utils.validation import InputValidator
        
        assert InputValidator.validate_concurrency(1) == 1
        assert InputValidator.validate_concurrency(50) == 50
        assert InputValidator.validate_concurrency(10, min_value=5, max_value=20) == 10
    
    def test_validate_concurrency_out_of_range(self):
        """Test rejecting concurrency out of range."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_concurrency(0, min_value=1)
        
        with pytest.raises(ValidationError):
            InputValidator.validate_concurrency(101, max_value=100)
    
    def test_validate_batch_size_valid(self):
        """Test validating a valid batch size."""
        from translatorhoi4.utils.validation import InputValidator
        
        assert InputValidator.validate_batch_size(50) == 50
        assert InputValidator.validate_batch_size(1) == 1
        assert InputValidator.validate_batch_size(500) == 500
    
    def test_validate_batch_size_out_of_range(self):
        """Test rejecting batch size out of range."""
        from translatorhoi4.utils.validation import InputValidator, ValidationError
        
        with pytest.raises(ValidationError):
            InputValidator.validate_batch_size(0)
        
        with pytest.raises(ValidationError):
            InputValidator.validate_batch_size(501)


class TestValidateSettings:
    """Test the validate_settings function."""
    
    def test_validate_settings_minimal(self, tmp_path):
        """Test validating minimal settings."""
        from translatorhoi4.utils.validation import validate_settings
        
        # Create a directory for testing
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        settings = {
            "src": str(src_dir),
        }
        
        result = validate_settings(settings)
        assert result["src"] == src_dir
    
    def test_validate_settings_with_languages(self, tmp_path):
        """Test validating settings with languages."""
        from translatorhoi4.utils.validation import validate_settings
        
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        settings = {
            "src": str(src_dir),
            "src_lang": "english",
            "dst_lang": "russian",
        }
        
        result = validate_settings(settings)
        assert result["src_lang"] == "english"
        assert result["dst_lang"] == "russian"
    
    def test_validate_settings_invalid_language(self, tmp_path):
        """Test rejecting invalid language in settings."""
        from translatorhoi4.utils.validation import validate_settings, ValidationError
        
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        settings = {
            "src": str(src_dir),
            "src_lang": "invalid",
        }
        
        with pytest.raises(ValidationError):
            validate_settings(settings)
