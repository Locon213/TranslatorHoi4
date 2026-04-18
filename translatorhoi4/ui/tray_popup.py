"""Custom tray popup widget with richer status/actions."""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from qfluentwidgets import BodyLabel, PrimaryPushButton, PushButton, StrongBodyLabel


class TrayPopup(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("trayPopup")
        self.setMinimumWidth(320)
        self.setStyleSheet(
            """
            QFrame#trayPopup {
                background: #20242b;
                border: 1px solid #343b46;
                border-radius: 16px;
            }
            QLabel {
                color: #f3f4f6;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.title_label = StrongBodyLabel("TranslatorHoi4", self)
        self.status_label = BodyLabel("Ready", self)
        self.file_label = BodyLabel("No active file", self)
        self.progress_label = BodyLabel("Progress: 0%", self)
        self.update_label = BodyLabel("Updates: idle", self)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.update_label)

        button_row_1 = QHBoxLayout()
        button_row_1.setSpacing(8)
        self.btn_show = PrimaryPushButton("Show", self)
        self.btn_pause = PushButton("Pause", self)
        self.btn_cancel = PushButton("Cancel", self)
        button_row_1.addWidget(self.btn_show)
        button_row_1.addWidget(self.btn_pause)
        button_row_1.addWidget(self.btn_cancel)

        button_row_2 = QHBoxLayout()
        button_row_2.setSpacing(8)
        self.btn_about = PushButton("About", self)
        self.btn_quit = PushButton("Quit", self)
        button_row_2.addWidget(self.btn_about)
        button_row_2.addStretch(1)
        button_row_2.addWidget(self.btn_quit)

        layout.addLayout(button_row_1)
        layout.addLayout(button_row_2)

        self._bind_button(self.btn_show, None)
        self._bind_button(self.btn_pause, None)
        self._bind_button(self.btn_cancel, None)
        self._bind_button(self.btn_about, None)
        self._bind_button(self.btn_quit, None)

    def _bind_button(self, button: PushButton, action: QAction | None) -> None:
        if action is None:
            button.setEnabled(False)
            return

        def sync() -> None:
            button.setText(action.text())
            button.setEnabled(action.isEnabled())

        action.changed.connect(sync)
        button.clicked.connect(action.trigger)
        sync()

    def bind_actions(
        self,
        *,
        show_action: QAction,
        pause_action: QAction,
        cancel_action: QAction,
        about_action: QAction,
        quit_action: QAction,
    ) -> None:
        self._bind_button(self.btn_show, show_action)
        self._bind_button(self.btn_pause, pause_action)
        self._bind_button(self.btn_cancel, cancel_action)
        self._bind_button(self.btn_about, about_action)
        self._bind_button(self.btn_quit, quit_action)

    def update_status(self, *, status: str, file_text: str, progress_text: str, update_text: str) -> None:
        self.status_label.setText(status)
        self.file_label.setText(file_text)
        self.progress_label.setText(progress_text)
        self.update_label.setText(update_text)

    def toggle_near_cursor(self) -> None:
        if self.isVisible():
            self.hide()
            return
        cursor_pos = QCursor.pos()
        self.adjustSize()
        position = QPoint(cursor_pos.x() - self.width() + 16, cursor_pos.y() - self.height() - 8)
        self.move(position)
        self.show()
        self.raise_()
        self.activateWindow()
