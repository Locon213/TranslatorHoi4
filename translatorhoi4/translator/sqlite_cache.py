"""SQLite-based disk cache for better performance with large projects."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


class SQLiteCache:
    """High-performance SQLite-based cache implementation.
    
    Features:
    - Thread-safe operations
    - Configurable cache size limits
    - Efficient bulk operations
    - LRU eviction policy
    - Optional compression for large entries
    """
    
    # SQL statements for creating the cache table
    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            accessed_at REAL DEFAULT (strftime('%s', 'now')),
            hit_count INTEGER DEFAULT 0,
            size_bytes INTEGER DEFAULT 0
        )
    """
    
    _CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache(accessed_at)
    """
    
    _CREATE_HIT_COUNT_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_hit_count ON cache(hit_count DESC)
    """
    
    def __init__(
        self,
        path: str,
        max_entries: int = 100000,
        max_memory_entries: int = 5000,
        compression_threshold: int = 1000,
    ):
        """Initialize the SQLite cache.
        
        Args:
            path: Path to the SQLite database file
            max_entries: Maximum number of entries in the cache
            max_memory_entries: Maximum number of entries to keep in memory
            compression_threshold: Minimum size in bytes to compress entries
        """
        self.path = path
        self.max_entries = max_entries
        self.max_memory_entries = max_memory_entries
        self.compression_threshold = compression_threshold
        self._memory_cache: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None
        self._initialized = False
        
        # Compression support
        try:
            import zlib
            self._compress = zlib.compress
            self._decompress = zlib.decompress
            self._has_compression = True
        except ImportError:
            self._has_compression = False
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with self._lock:
            if self._initialized:
                return
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            
            # Connect to database
            self._db = sqlite3.connect(
                self.path,
                check_same_thread=False,
                isolation_level=None,
            )
            
            # Enable WAL mode for better concurrent access
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA cache_size=-64000")  # 64MB cache
            
            # Create table and indexes
            self._db.execute(self._CREATE_TABLE_SQL)
            self._db.execute(self._CREATE_INDEX_SQL)
            self._db.execute(self._CREATE_HIT_COUNT_INDEX_SQL)
            
            self._initialized = True
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper initialization."""
        if not self._initialized:
            self._init_db()
        yield self._db
    
    def _encode_value(self, value: str) -> bytes:
        """Encode and optionally compress a value."""
        if not self._has_compression:
            return value.encode("utf-8")
        
        encoded = value.encode("utf-8")
        if len(encoded) >= self.compression_threshold:
            return b"\x01" + self._compress(encoded)
        return b"\x00" + encoded
    
    def _decode_value(self, data: bytes) -> str:
        """Decode and optionally decompress a value."""
        if not data:
            return ""
        
        if data[0:1] == b"\x01" and self._has_compression:
            return self._decompress(data[1:]).decode("utf-8")
        elif data[0:1] == b"\x00":
            return data[1:].decode("utf-8")
        else:
            return data.decode("utf-8")
    
    def get(self, key: str) -> Optional[str]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                value, timestamp = self._memory_cache[key]
                # Update access order
                del self._memory_cache[key]
                self._memory_cache[key] = (value, timestamp)
                return value
            
            try:
                with self._get_connection() as db:
                    cursor = db.execute(
                        "SELECT value, accessed_at FROM cache WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        value = self._decode_value(row[0])
                        accessed_at = row[1]
                        
                        # Update access time and hit count
                        db.execute(
                            "UPDATE cache SET accessed_at = strftime('%s', 'now'), hit_count = hit_count + 1 WHERE key = ?",
                            (key,)
                        )
                        
                        # Add to memory cache
                        self._add_to_memory(key, value, accessed_at)
                        
                        return value
            except Exception:
                pass
        
        return None
    
    def set(self, key: str, value: str):
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Add to memory cache
            self._add_to_memory(key, value)
            
            try:
                with self._get_connection() as db:
                    encoded = self._encode_value(value)
                    db.execute(
                        """INSERT OR REPLACE INTO cache (key, value, created_at, accessed_at, size_bytes)
                           VALUES (?, ?, strftime('%s', 'now'), strftime('%s', 'now'), ?)
                        """,
                        (key, encoded, len(encoded))
                    )
                    
                    # Evict old entries if needed
                    self._evict_if_needed(db)
                    
            except Exception:
                pass
    
    def _add_to_memory(self, key: str, value: str, timestamp: Optional[float] = None):
        """Add entry to memory cache with LRU eviction."""
        import time
        if timestamp is None:
            timestamp = time.time()
        
        # Evict oldest entry if at capacity
        while len(self._memory_cache) >= self.max_memory_entries:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = (value, timestamp)
    
    def _evict_if_needed(self, db: sqlite3.Connection):
        """Evict entries if cache is at capacity."""
        # Check entry count
        cursor = db.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        
        if count > self.max_entries:
            # Evict by access time (oldest first)
            db.execute(
                """DELETE FROM cache WHERE key IN (
                   SELECT key FROM cache ORDER BY accessed_at ASC LIMIT ?
                )""",
                (count - self.max_entries + 1000,)
            )
        
        # Check total size
        cursor = db.execute("SELECT SUM(size_bytes) FROM cache")
        total_size = cursor.fetchone()[0] or 0
        max_size_bytes = self.max_entries * 500  # Assume average 500 bytes per entry
        
        if total_size > max_size_bytes:
            # Evict by hit count (lowest first)
            db.execute(
                """DELETE FROM cache WHERE key IN (
                   SELECT key FROM cache ORDER BY hit_count ASC, accessed_at ASC LIMIT ?
                )""",
                (int(total_size - max_size_bytes) // 500 + 1000,)
            )
    
    def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            # Remove from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]
            
            try:
                with self._get_connection() as db:
                    cursor = db.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return cursor.rowcount > 0
            except Exception:
                pass
        
        return False
    
    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            self._memory_cache.clear()
            
            try:
                with self._get_connection() as db:
                    db.execute("DELETE FROM cache")
                    db.execute("VACUUM")
            except Exception:
                pass
    
    def clear_pattern(self, pattern: str):
        """Clear all entries matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.yml")
        """
        import fnmatch
        
        with self._lock:
            # Get all keys matching pattern
            keys_to_delete = []
            for key in list(self._memory_cache.keys()):
                if fnmatch.fnmatch(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._memory_cache[key]
            
            try:
                with self._get_connection() as db:
                    # SQLite doesn't support LIKE with wildcards in DELETE directly
                    # so we need to fetch matching keys first
                    cursor = db.execute("SELECT key FROM cache")
                    keys = [row[0] for row in cursor.fetchall()]
                    
                    for key in keys:
                        if fnmatch.fnmatch(key, pattern):
                            db.execute("DELETE FROM cache WHERE key = ?", (key,))
            except Exception:
                pass
    
    def get_many(self, keys: List[str]) -> Dict[str, str]:
        """Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of found key-value pairs
        """
        result = {}
        
        with self._lock:
            # Check memory cache first
            keys_to_fetch = []
            for key in keys:
                if key in self._memory_cache:
                    value, _ = self._memory_cache[key]
                    result[key] = value
                else:
                    keys_to_fetch.append(key)
            
            if not keys_to_fetch:
                return result
            
            try:
                with self._get_connection() as db:
                    placeholders = ",".join("?" * len(keys_to_fetch))
                    cursor = db.execute(
                        f"SELECT key, value FROM cache WHERE key IN ({placeholders})",
                        keys_to_fetch
                    )
                    
                    for key, encoded in cursor.fetchall():
                        value = self._decode_value(encoded)
                        result[key] = value
                        
                        # Update access time
                        db.execute(
                            "UPDATE cache SET accessed_at = strftime('%s', 'now'), hit_count = hit_count + 1 WHERE key = ?",
                            (key,)
                        )
                        
                        # Add to memory cache
                        self._add_to_memory(key, value)
                        
            except Exception:
                pass
        
        return result
    
    def set_many(self, entries: Dict[str, str]):
        """Set multiple values in the cache.
        
        Args:
            entries: Dictionary of key-value pairs
        """
        with self._lock:
            # Add to memory cache
            for key, value in entries.items():
                self._add_to_memory(key, value)
            
            try:
                with self._get_connection() as db:
                    encoded_entries = [
                        (key, self._encode_value(value))
                        for key, value in entries.items()
                    ]
                    
                    db.executemany(
                        """INSERT OR REPLACE INTO cache (key, value, created_at, accessed_at, size_bytes)
                           VALUES (?, ?, strftime('%s', 'now'), strftime('%s', 'now'), ?)
                        """,
                        [(key, encoded, len(encoded)) for key, encoded in encoded_entries]
                    )
                    
                    # Evict old entries if needed
                    self._evict_if_needed(db)
                    
            except Exception:
                pass
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            stats = {
                "memory_entries": len(self._memory_cache),
                "max_memory_entries": self.max_memory_entries,
            }
            
            try:
                with self._get_connection() as db:
                    cursor = db.execute("SELECT COUNT(*) FROM cache")
                    stats["total_entries"] = cursor.fetchone()[0]
                    
                    cursor = db.execute("SELECT SUM(size_bytes) FROM cache")
                    total_size = cursor.fetchone()[0] or 0
                    stats["total_size_bytes"] = total_size
                    stats["total_size_human"] = self._human_size(total_size)
                    
                    cursor = db.execute("SELECT MAX(hit_count) FROM cache")
                    stats["max_hits"] = cursor.fetchone()[0] or 0
                    
                    cursor = db.execute("SELECT AVG(hit_count) FROM cache")
                    stats["avg_hits"] = cursor.fetchone()[0] or 0
                    
            except Exception:
                stats["total_entries"] = 0
                stats["total_size_bytes"] = 0
                stats["total_size_human"] = "0 B"
                stats["max_hits"] = 0
                stats["avg_hits"] = 0
            
            return stats
    
    def _human_size(self, size: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def optimize(self):
        """Optimize the database for better performance."""
        try:
            with self._get_connection() as db:
                db.execute("VACUUM")
                db.execute("PRAGMA optimize")
        except Exception:
            pass
    
    def close(self):
        """Close the database connection."""
        with self._lock:
            if self._db:
                self._db.close()
                self._db = None
                self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# --- Cache Factory ---
class CacheFactory:
    """Factory for creating cache instances with different backends."""
    
    _instance: Optional["CacheFactory"] = None
    
    def __new__(cls) -> "CacheFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def create_cache(
        self,
        cache_type: str = "auto",
        path: Optional[str] = None,
        max_entries: int = 100000,
        **kwargs
    ):
        """Create a cache instance.
        
        Args:
            cache_type: Type of cache ("sqlite", "json", "auto")
            path: Path to cache file/database
            max_entries: Maximum number of entries
            **kwargs: Additional arguments for the cache
            
        Returns:
            Cache instance
        """
        if cache_type == "auto":
            # Auto-detect based on expected size
            if max_entries > 50000:
                cache_type = "sqlite"
            else:
                cache_type = "json"
        
        if cache_type == "sqlite":
            if path is None:
                path = ".translatorhoi4_cache.db"
            return SQLiteCache(path, max_entries, **kwargs)
        else:
            # Fall back to JSON-based cache
            from .cache import DiskCache
            if path is None:
                path = ".translatorhoi4_cache.json"
            return DiskCache(path, max_mem=min(max_entries, 50000))


# Global cache factory
cache_factory = CacheFactory()
