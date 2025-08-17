"""Application bootstrap."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

try:
    from translatorhoi4.ui.main_window import MainWindow
except Exception:
    from .ui.main_window import MainWindow  

def _install_excepthook() -> None:
    base = Path(sys.argv[0]).resolve().parent
    def handle(exc_type, exc, tb):
        log = base / "error.log"
        with log.open("w", encoding="utf-8") as fh:
            traceback.print_exception(exc_type, exc, tb, file=fh)
    sys.excepthook = handle

def _res_path(rel: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / rel)
    return str(Path(__file__).resolve().parent.parent / rel)

def main(argv: list[str] | None = None) -> None:
    _install_excepthook()
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run offscreen and exit")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1] + (argv or []))
    app.setWindowIcon(QIcon(_res_path("assets/icon.png")))
    w = MainWindow()
    if args.smoke:
        w.move(-10000, -10000)
        w.show()
        QTimer.singleShot(100, app.quit)
        app.exec()
        return
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
