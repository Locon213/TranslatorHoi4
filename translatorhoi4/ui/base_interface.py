"""Base scrollable interface class used by all UI pages."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import ScrollArea


class BaseInterface(ScrollArea):
    """Base class for pages to provide scrolling."""
    def __init__(self, objectName, parent=None):
        super().__init__(parent)
        self.setObjectName(objectName)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)
        self.vBoxLayout.setContentsMargins(30, 20, 30, 20)
        self.vBoxLayout.setSpacing(15)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea {border: none; background: transparent}")
        self.view.setStyleSheet("QWidget {background: transparent}")
