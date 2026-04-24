"""Custom tray popup widget with richer status/actions."""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from qfluentwidgets import BodyLabel, CaptionLabel, CardWidget, FluentIcon as FIF, PrimaryPushButton, PushButton, StrongBodyLabel


class TrayPopup(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("trayPopup")
        self.setMinimumWidth(360)
        self.setStyleSheet(
            """
            QFrame#trayPopup {
                background: #11141a;
                border: 1px solid #303846;
                border-radius: 16px;
            }
            QWidget#trayHeader {
                background: transparent;
            }
            QLabel#trayAccentDot {
                background: #22d3ee;
                border-radius: 5px;
            }
            QLabel#trayTitle {
                color: #f6f8fb;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#traySubtitle,
            QLabel#metricName {
                color: #9aa7b7;
            }
            QLabel#metricValue {
                color: #edf2f8;
                font-weight: 600;
            }
            QWidget#statusCard {
                background: #1b2029;
                border: 1px solid #303846;
                border-radius: 8px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        header = QWidget(self)
        header.setObjectName("trayHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        self.accent_dot = QLabel(header)
        self.accent_dot.setObjectName("trayAccentDot")
        self.accent_dot.setFixedSize(10, 10)

        title_block = QWidget(header)
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        self.title_label = StrongBodyLabel("TranslatorHoi4", title_block)
        self.title_label.setObjectName("trayTitle")
        self.subtitle_label = CaptionLabel("Translator status", title_block)
        self.subtitle_label.setObjectName("traySubtitle")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)

        header_layout.addWidget(self.accent_dot)
        header_layout.addWidget(title_block, 1)

        status_card = CardWidget(self)
        status_card.setObjectName("statusCard")
        status_layout = QGridLayout(status_card)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setHorizontalSpacing(16)
        status_layout.setVerticalSpacing(8)

        self.status_label = BodyLabel("Ready", status_card)
        self.file_label = BodyLabel("No active file", status_card)
        self.progress_label = BodyLabel("0%", status_card)
        self.update_label = BodyLabel("Idle", status_card)

        for row, (name, value) in enumerate(
            [
                ("Status", self.status_label),
                ("File", self.file_label),
                ("Progress", self.progress_label),
                ("Updates", self.update_label),
            ]
        ):
            name_label = CaptionLabel(name, status_card)
            name_label.setObjectName("metricName")
            value.setObjectName("metricValue")
            value.setWordWrap(True)
            status_layout.addWidget(name_label, row, 0)
            status_layout.addWidget(value, row, 1)

        layout.addWidget(header)
        layout.addWidget(status_card)

        button_row_1 = QHBoxLayout()
        button_row_1.setSpacing(8)
        self.btn_show = PrimaryPushButton("Show Window", self, FIF.VIEW)
        self.btn_pause = PushButton("Pause", self, FIF.PAUSE)
        self.btn_cancel = PushButton("Cancel", self, FIF.CANCEL)
        self.btn_show.setMinimumWidth(132)
        self.btn_pause.setMinimumWidth(86)
        self.btn_cancel.setMinimumWidth(92)
        button_row_1.addWidget(self.btn_show)
        button_row_1.addWidget(self.btn_pause)
        button_row_1.addWidget(self.btn_cancel)

        button_row_2 = QHBoxLayout()
        button_row_2.setSpacing(8)
        self.btn_about = PushButton("About", self, FIF.INFO)
        self.btn_quit = PushButton("Quit", self, FIF.POWER_BUTTON)
        self.btn_about.setMinimumWidth(92)
        self.btn_quit.setMinimumWidth(84)
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
        self.progress_label.setText(progress_text.replace("Progress: ", ""))
        self.update_label.setText(update_text.replace("Updates: ", ""))
        self._sync_accent(status=status, update_text=update_text)

    def _sync_accent(self, *, status: str, update_text: str) -> None:
        status_lower = status.lower()
        update_lower = update_text.lower()
        if "error" in status_lower or "failed" in status_lower:
            color = "#f87171"
        elif "translat" in status_lower or "running" in status_lower:
            color = "#22d3ee"
        elif "available" in update_lower:
            color = "#facc15"
        else:
            color = "#34d399"
        self.accent_dot.setStyleSheet(
            f"QLabel#trayAccentDot {{ background: {color}; border-radius: 5px; }}"
        )

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
