"""About dialog."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, QSize
from PyQt6.QtGui import QDesktopServices, QIcon
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout

from qfluentwidgets import (
    PrimaryPushButton, 
    HyperlinkButton,
    TitleLabel, 
    BodyLabel, 
    ImageLabel,
    FluentIcon as FIF,
    setTheme, 
    Theme
)

from ..utils.version import __version__


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Принудительно ставим темную тему
        setTheme(Theme.DARK)
        
        self.setWindowTitle("About TranslatorHoi4")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Фиксированный размер и цвет фона
        self.setFixedSize(400, 380)
        self.setStyleSheet("QDialog { background-color: rgb(32, 32, 32); }")
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.setContentsMargins(30, 40, 30, 30)
        layout.setSpacing(10)

        # 1. Логотип (ImageLabel из qfluentwidgets)
        self.icon_label = ImageLabel("assets/icon.png", self)
        self.icon_label.setFixedSize(100, 100)
        self.icon_label.setScaledContents(True)
        # Исправляем ошибку: указываем радиус для всех 4 углов (TL, TR, BL, BR)
        self.icon_label.setBorderRadius(20, 20, 20, 20)
        
        # 2. Название приложения (Крупный шрифт)
        self.title_label = TitleLabel("TranslatorHoi4", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 3. Версия
        self.version_label = BodyLabel(f"Version {__version__}", self)
        self.version_label.setStyleSheet("color: #d1d1d1;") # Чуть более тусклый белый

        # 4. Автор
        self.author_label = BodyLabel("Developed by Locon213", self)
        self.author_label.setStyleSheet("color: #d1d1d1;")

        # 5. Ссылка на GitHub (HyperlinkButton с иконкой)
        self.github_link = HyperlinkButton(
            url="https://github.com/Locon213/TranslatorHoi4",
            text="GitHub Repository",
            parent=self,
            icon=FIF.GITHUB
        )

        # 6. Кнопка OK внизу
        self.ok_btn = PrimaryPushButton("Close", self)
        self.ok_btn.setFixedWidth(140)
        self.ok_btn.clicked.connect(self.accept)

        # Добавляем всё в Layout
        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(10)
        layout.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.version_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.author_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(10)
        layout.addWidget(self.github_link, 0, Qt.AlignmentFlag.AlignHCenter)
        
        layout.addStretch(1) # Растяжка, чтобы кнопка уехала вниз
        layout.addWidget(self.ok_btn, 0, Qt.AlignmentFlag.AlignHCenter)