"""Disk cache implementation (ported from monolith)."""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Dict, Optional


class DiskCache:
    def __init__(self, path: str, max_mem: int = 50000):
        self.path = path
        self.max_mem = max_mem
        self.mem = OrderedDict()
        self._file_map: Dict[str, str] = {}
        self._loaded = False
        self._lock = threading.RLock()

    def load(self):
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not os.path.isfile(self.path):
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._file_map = json.load(f)
            except Exception:
                self._file_map = {}

    def get(self, k: str) -> Optional[str]:
        with self._lock:
            if not self._loaded:
                self.load()
            if k in self.mem:
                v = self.mem.pop(k)
                self.mem[k] = v
                return v
            v = self._file_map.get(k)
            if v is not None:
                self.mem[k] = v
                self._trim()
            return v

    def set(self, k: str, v: str):
        with self._lock:
            if not self._loaded:
                self.load()
            self._file_map[k] = v
            self.mem[k] = v
            self._trim()

    def _trim(self):
        while len(self.mem) > self.max_mem:
            self.mem.popitem(last=False)

    def save(self):
        with self._lock:
            if not self._loaded:
                return
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self._file_map, f, ensure_ascii=False)
            except Exception:
                pass
