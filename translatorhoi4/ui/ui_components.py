"""Custom UI components for the main window."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

from qfluentwidgets import SimpleCardWidget, BodyLabel, SubtitleLabel


class SettingCard(SimpleCardWidget):
    """Helper to create a nice looking card for settings."""
    def __init__(self, title, widget, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 10, 20, 10)

        self.lbl_title = BodyLabel(title, self)
        self.layout.addWidget(self.lbl_title)
        self.layout.addStretch(1)
        self.layout.addWidget(widget)


class SectionHeader(QWidget):
    """Simple bold header for sections."""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 5)
        self.lbl = SubtitleLabel(text, self)
        layout.addWidget(self.lbl)


class LoadingIndicator(QWidget):
    """Simple spinning arc indicator."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setFixedSize(16, 16)

    def start(self):
        self._timer.start(100)
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _advance(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):
        from qfluentwidgets import Theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        pen = QPen(QColor(255, 255, 255) if Theme.DARK else QColor(0, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawArc(-6, -6, 12, 12, 0, 270 * 16)