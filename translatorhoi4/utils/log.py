"""Logging helper."""
from __future__ import annotations

import sys


def log(message: str) -> None:
    sys.stderr.write(message + "\n")
