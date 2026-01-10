"""UI threads for background operations."""
from __future__ import annotations

import requests
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class IOModelFetchThread(QThread):
    """Thread for fetching available models from IO Intelligence API."""
    ready = pyqtSignal(list)
    fail = pyqtSignal(str)

    def __init__(self, api_key: Optional[str], base_url: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url

    def run(self):
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = requests.get(self.base_url.rstrip('/') + "/models", headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
            self.ready.emit(models)
        except Exception as e:
            self.fail.emit(str(e))