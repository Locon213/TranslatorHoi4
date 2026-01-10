"""Tests for env.py - Environment variable handling."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


class TestEnvLoader:
    """Test environment variable loading."""
    
    def test_env_get_with_default(self, monkeypatch, tmp_path):
        """Test getting environment variable with default."""
        # Clear any existing env var
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        
        from translatorhoi4.utils.env import env
        
        result = env.get("TEST_API_KEY", "default_value")
        assert result == "default_value"
    
    def test_env_set_and_get(self, monkeypatch):
        """Test setting and getting environment variable."""
        from translatorhoi4.utils.env import env
        
        env.set("TEST_VAR", "test_value")
        result = env.get("TEST_VAR")
        assert result == "test_value"
        
        # Cleanup
        monkeypatch.delenv("TEST_VAR", raising=False)
    
    def test_env_get_bool(self, monkeypatch):
        """Test getting boolean environment variable."""
        from translatorhoi4.utils.env import env
        
        monkeypatch.setenv("TEST_BOOL_TRUE", "true")
        monkeypatch.setenv("TEST_BOOL_FALSE", "false")
        monkeypatch.setenv("TEST_BOOL_1", "1")
        monkeypatch.setenv("TEST_BOOL_EMPTY", "")
        
        assert env.get_bool("TEST_BOOL_TRUE") is True
        assert env.get_bool("TEST_BOOL_FALSE") is False
        assert env.get_bool("TEST_BOOL_1") is True
        assert env.get_bool("TEST_BOOL_EMPTY") is False
        assert env.get_bool("TEST_NONEXISTENT") is False
        
        # Cleanup
        monkeypatch.delenv("TEST_BOOL_TRUE", raising=False)
        monkeypatch.delenv("TEST_BOOL_FALSE", raising=False)
        monkeypatch.delenv("TEST_BOOL_1", raising=False)
        monkeypatch.delenv("TEST_BOOL_EMPTY", raising=False)
    
    def test_env_get_int(self, monkeypatch):
        """Test getting integer environment variable."""
        from translatorhoi4.utils.env import env
        
        monkeypatch.setenv("TEST_INT", "42")
        monkeypatch.setenv("TEST_FLOAT_AS_INT", "3.14")
        
        assert env.get_int("TEST_INT") == 42
        assert env.get_int("TEST_FLOAT_AS_INT") == 3  # Truncated
        assert env.get_int("TEST_NONEXISTENT", default=10) == 10
        assert env.get_int("TEST_NONEXISTENT") == 0
        
        # Cleanup
        monkeypatch.delenv("TEST_INT", raising=False)
        monkeypatch.delenv("TEST_FLOAT_AS_INT", raising=False)
    
    def test_env_get_float(self, monkeypatch):
        """Test getting float environment variable."""
        from translatorhoi4.utils.env import env
        
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        monkeypatch.setenv("TEST_INT_AS_FLOAT", "42")
        
        assert env.get_float("TEST_FLOAT") == 3.14
        assert env.get_float("TEST_INT_AS_FLOAT") == 42.0
        assert env.get_float("TEST_NONEXISTENT", default=1.5) == 1.5
        assert env.get_float("TEST_NONEXISTENT") == 0.0
        
        # Cleanup
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        monkeypatch.delenv("TEST_INT_AS_FLOAT", raising=False)


class TestAPIKeyHelpers:
    """Test API key helper functions."""
    
    def test_get_api_key_not_found(self, monkeypatch):
        """Test getting API key when not set."""
        from translatorhoi4.utils.env import get_api_key
        
        # Ensure no API keys are set
        for key in ["G4F_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        
        result = get_api_key("g4f")
        assert result is None
    
    def test_set_api_key(self, monkeypatch):
        """Test setting API key."""
        from translatorhoi4.utils.env import get_api_key, set_api_key
        
        set_api_key("g4f", "test_key_123")
        result = get_api_key("g4f")
        assert result == "test_key_123"
        
        # Cleanup
        monkeypatch.delenv("G4F_API_KEY", raising=False)
