"""Tests for sqlite_cache.py - SQLite-based disk cache."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestSQLiteCache:
    """Test SQLite cache functionality."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create a SQLite cache instance for testing."""
        from translatorhoi4.translator.sqlite_cache import SQLiteCache
        
        cache_path = tmp_path / "test_cache.db"
        cache = SQLiteCache(
            str(cache_path),
            max_entries=1000,
            max_memory_entries=100,
        )
        yield cache
        cache.close()
    
    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
    
    def test_get_nonexistent(self, cache):
        """Test getting a nonexistent key returns None."""
        result = cache.get("nonexistent")
        assert result is None
    
    def test_delete(self, cache):
        """Test deleting a key."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None
    
    def test_delete_nonexistent(self, cache):
        """Test deleting a nonexistent key returns False."""
        result = cache.delete("nonexistent")
        assert result is False
    
    def test_clear(self, cache):
        """Test clearing the cache."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None
    
    def test_get_many(self, cache):
        """Test getting multiple keys at once."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        result = cache.get_many(["key1", "key2", "nonexistent"])
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert "nonexistent" not in result
    
    def test_set_many(self, cache):
        """Test setting multiple keys at once."""
        entries = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
        
        cache.set_many(entries)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.stats()
        assert stats["memory_entries"] >= 0
        assert stats["total_entries"] >= 2
    
    def test_unicode_support(self, cache):
        """Test storing and retrieving Unicode content."""
        unicode_value = "Привет мир! 🌍 Hello World!"
        cache.set("unicode_key", unicode_value)
        result = cache.get("unicode_key")
        assert result == unicode_value
    
    def test_large_value(self, cache):
        """Test storing large values."""
        large_value = "x" * 10000
        cache.set("large_key", large_value)
        result = cache.get("large_key")
        assert result == large_value
    
    def test_special_characters(self, cache):
        """Test storing values with special characters."""
        special_value = "Line1\\nLine2\tTab\r\n\"Quotes\" 'Single'"
        cache.set("special_key", special_value)
        result = cache.get("special_key")
        assert result == special_value
    
    def test_context_manager(self, tmp_path):
        """Test using cache as context manager."""
        from translatorhoi4.translator.sqlite_cache import SQLiteCache
        
        cache_path = tmp_path / "context_test.db"
        with SQLiteCache(str(cache_path)) as cache:
            cache.set("key", "value")
            result = cache.get("key")
            assert result == "value"
        
        # Should be closed after context
        assert cache._db is None
    
    def test_optimize(self, cache):
        """Test cache optimization."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Should not raise an exception
        cache.optimize()


class TestCacheFactory:
    """Test cache factory functionality."""
    
    def test_create_sqlite_cache(self, tmp_path):
        """Test creating SQLite cache via factory."""
        from translatorhoi4.translator.sqlite_cache import cache_factory
        
        cache = cache_factory.create_cache(
            cache_type="sqlite",
            path=str(tmp_path / "factory_test.db"),
        )
        assert cache is not None
        cache.close()
    
    def test_create_json_cache(self, tmp_path):
        """Test creating JSON cache via factory."""
        from translatorhoi4.translator.sqlite_cache import cache_factory
        
        cache = cache_factory.create_cache(
            cache_type="json",
            path=str(tmp_path / "json_test.json"),
        )
        assert cache is not None
