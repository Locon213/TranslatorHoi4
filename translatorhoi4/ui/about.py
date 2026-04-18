"""About dialog with update status and actions."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QHBoxLayout, QTextEdit, QVBoxLayout

from qfluentwidgets import (
    BodyLabel,
    HyperlinkButton,
    ImageLabel,
    PrimaryPushButton,
    PushButton,
    TitleLabel,
)

from ..utils.update_checker import check_for_updates, download_and_open_update
from ..utils.version import get_version_info


class _UpdateCheckThread(QThread):
    finished_info = Signal(dict)

    def __init__(self, force: bool = False) -> None:
        super().__init__()
        self.force = force

    def run(self) -> None:
        self.finished_info.emit(check_for_updates(force=self.force))


class _UpdateDownloadThread(QThread):
    finished_info = Signal(dict)
    failed = Signal(str)

    def run(self) -> None:
        try:
            self.finished_info.emit(download_and_open_update(force_check=False))
        except Exception as exc:
            self.failed.emit(str(exc))


class AboutDialog(QDialog):
    def __init__(self, parent=None, update_info: dict | None = None):
        super().__init__(parent)
        self._update_thread: _UpdateCheckThread | None = None
        self._download_thread: _UpdateDownloadThread | None = None
        self._update_info = update_info or {}
        self._version_info = get_version_info()

        self.setWindowTitle("About TranslatorHoi4")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(520, 460)
        self._setup_ui()
        self._render_update_info(self._update_info)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.icon_label = ImageLabel("assets/icon.png", self)
        self.icon_label.setFixedSize(84, 84)
        self.icon_label.setScaledContents(True)
        self.icon_label.setBorderRadius(18, 18, 18, 18)

        self.title_label = TitleLabel("TranslatorHoi4", self)
        self.version_label = BodyLabel(
            f"Version {self._version_info['version']} ({self._version_info['platform']}/{self._version_info['arch']})",
            self,
        )
        self.channel_label = BodyLabel(f"Channel: {self._version_info['channel']}", self)
        self.author_label = BodyLabel("Developed by Locon213", self)
        self.status_label = BodyLabel("Update status: not checked yet", self)

        self.release_notes = QTextEdit(self)
        self.release_notes.setReadOnly(True)
        self.release_notes.setPlaceholderText("Release notes will appear here.")
        self.release_notes.setMinimumHeight(180)

        self.btn_check_updates = PushButton("Check for Updates", self)
        self.btn_check_updates.clicked.connect(self._check_updates)

        self.btn_download_update = PrimaryPushButton("Download Update", self)
        self.btn_download_update.setEnabled(False)
        self.btn_download_update.clicked.connect(self._download_update)

        self.github_link = HyperlinkButton(
            url="https://github.com/Locon213/TranslatorHoi4",
            text="GitHub Releases",
            parent=self,
        )
        self.ok_btn = PrimaryPushButton("Close", self)
        self.ok_btn.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.btn_check_updates)
        buttons.addWidget(self.btn_download_update)
        buttons.addStretch(1)
        buttons.addWidget(self.github_link)
        buttons.addWidget(self.ok_btn)

        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.version_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.channel_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.author_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.status_label)
        layout.addWidget(self.release_notes)
        layout.addLayout(buttons)

    def _render_update_info(self, update_info: dict) -> None:
        self._update_info = update_info or {}
        if self._update_info.get("error"):
            self.status_label.setText(f"Update status: check failed ({self._update_info['error']})")
        elif self._update_info.get("update_available"):
            version = self._update_info.get("latest_version", "unknown")
            asset_name = self._update_info.get("asset_name", "matching asset")
            self.status_label.setText(f"Update available: {version} ({asset_name})")
        elif self._update_info.get("latest_version"):
            self.status_label.setText(f"Up to date. Latest version: {self._update_info['latest_version']}")
        else:
            self.status_label.setText("Update status: not checked yet")

        notes = self._update_info.get("release_notes") or "No release notes available."
        self.release_notes.setPlainText(notes)
        self.btn_download_update.setEnabled(bool(self._update_info.get("update_available") and self._update_info.get("download_url")))

    def _check_updates(self, force: bool = True) -> None:
        if self._update_thread and self._update_thread.isRunning():
            return
        self.status_label.setText("Update status: checking...")
        self.btn_check_updates.setEnabled(False)
        self._update_thread = _UpdateCheckThread(force=force)
        self._update_thread.finished_info.connect(self._on_update_finished)
        self._update_thread.finished.connect(lambda: self.btn_check_updates.setEnabled(True))
        self._update_thread.start()

    def _on_update_finished(self, update_info: dict) -> None:
        self._render_update_info(update_info)

    def _download_update(self) -> None:
        if self._download_thread and self._download_thread.isRunning():
            return
        self.status_label.setText("Update status: downloading update package...")
        self.btn_download_update.setEnabled(False)
        self._download_thread = _UpdateDownloadThread()
        self._download_thread.finished_info.connect(self._on_download_finished)
        self._download_thread.failed.connect(self._on_download_failed)
        self._download_thread.start()

    def _on_download_finished(self, info: dict) -> None:
        version = info.get("latest_version", "unknown")
        message = info.get("message", "Update package prepared.")
        self.status_label.setText(f"Update {version}: {message}")
        self.btn_download_update.setEnabled(bool(self._update_info.get("download_url")))

    def _on_download_failed(self, error: str) -> None:
        self.status_label.setText(f"Update status: failed to download update ({error})")
        self.btn_download_update.setEnabled(bool(self._update_info.get("download_url")))
