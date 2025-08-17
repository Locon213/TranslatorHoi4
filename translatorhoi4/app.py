"""Application bootstrap."""
from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from .ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("assets/icon.png"))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
