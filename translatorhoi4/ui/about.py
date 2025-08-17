"""About dialog."""
from __future__ import annotations

from PyQt6.QtGui import QDesktopServices, QIcon
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

from ..utils.version import __version__
from .theme import DARK_QSS


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About TranslatorHoi4")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>TranslatorHoi4</b>"))
        layout.addWidget(QLabel(f"Version: {__version__}"))
        layout.addWidget(QLabel("Author: Locon213"))
        link = QLabel('<a href="https://github.com/Locon213/TranslatorHoi4">https://github.com/Locon213/TranslatorHoi4</a>')
        link.setOpenExternalLinks(True)
        layout.addWidget(link)
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_row.addStretch(1)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)
        self.setFixedSize(400, 200)
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setStyleSheet(DARK_QSS)
