"""About dialog with update status and actions."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QThread, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CaptionLabel,
    ComboBox,
    FluentIcon as FIF,
    HyperlinkButton,
    ImageLabel,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
    TextBrowser,
    TitleLabel,
)

from .language_options import populate_language_combo
from .translations import translate_text
from ..utils.update_checker import check_for_updates, download_and_open_update
from ..utils.version import get_version_info


def _get_resource_path(rel_path: str) -> str:
    """Resolve bundled resources in development and frozen builds."""
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent.parent
    return str(base_path / rel_path)


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
    language_changed = Signal(str)

    def __init__(self, parent=None, update_info: dict | None = None, current_language: str = "english"):
        super().__init__(parent)
        self._update_thread: _UpdateCheckThread | None = None
        self._download_thread: _UpdateDownloadThread | None = None
        self._update_info = update_info or {}
        self._version_info = get_version_info()
        self._lang_code = current_language or "english"
        self._drag_position: QPoint | None = None
        self._icon_path = _get_resource_path("assets/icon.png")

        self.setWindowTitle("About TranslatorHoi4")
        self.setObjectName("aboutDialog")
        self.setWindowIcon(QIcon(self._icon_path))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.resize(700, 560)
        self.setMinimumSize(660, 500)
        self._setup_ui()
        self._apply_translations(self._lang_code)
        self._render_update_info(self._update_info)

    def _setup_ui(self) -> None:
        self._apply_styles()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        header = QFrame(self)
        header.setObjectName("titleBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 8, 8)
        header_layout.setSpacing(8)

        window_icon = ImageLabel(self._icon_path, header)
        window_icon.setFixedSize(18, 18)
        window_icon.setScaledContents(True)

        self.window_title = CaptionLabel("About TranslatorHoi4", header)
        self.window_title.setObjectName("windowTitle")

        self.btn_title_close = PushButton("×", header)
        self.btn_title_close.setObjectName("titleCloseButton")
        self.btn_title_close.setFixedSize(34, 30)
        self.btn_title_close.setToolTip("Close")
        self.btn_title_close.clicked.connect(self.accept)

        header_layout.addWidget(window_icon)
        header_layout.addWidget(self.window_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.btn_title_close)

        content = QWidget(self)
        content.setObjectName("aboutContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 18, 24, 24)
        content_layout.setSpacing(14)

        hero = CardWidget(content)
        hero.setObjectName("heroPanel")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(18, 18, 18, 18)
        hero_layout.setSpacing(16)

        self.icon_label = ImageLabel(self._icon_path, hero)
        self.icon_label.setFixedSize(82, 82)
        self.icon_label.setObjectName("appIcon")
        self.icon_label.setScaledContents(True)
        self.icon_label.setBorderRadius(16, 16, 16, 16)

        title_block = QWidget(hero)
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(6)

        self.title_label = TitleLabel("TranslatorHoi4", title_block)
        self.title_label.setObjectName("appTitle")
        self.version_label = BodyLabel(
            f"Version {self._version_info['version']} ({self._version_info['platform']}/{self._version_info['arch']})",
            title_block,
        )
        self.channel_label = BodyLabel(f"Channel: {self._version_info['channel']}", title_block)
        self.author_label = BodyLabel("Developed by Locon213", title_block)
        self.version_label.setObjectName("mutedLabel")
        self.channel_label.setObjectName("mutedLabel")
        self.author_label.setObjectName("mutedLabel")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.version_label)
        title_layout.addWidget(self.channel_label)
        title_layout.addWidget(self.author_label)

        hero_layout.addWidget(self.icon_label)
        hero_layout.addWidget(title_block, 1)

        language_card = CardWidget(content)
        language_card.setObjectName("statusCard")
        language_layout = QHBoxLayout(language_card)
        language_layout.setContentsMargins(14, 12, 14, 12)
        language_layout.setSpacing(10)

        self.language_label = StrongBodyLabel("Interface Language", language_card)
        self.cmb_language = ComboBox(language_card)
        self.cmb_language.setMinimumWidth(220)
        populate_language_combo(self.cmb_language)
        self._set_language_combo(self._lang_code)
        self.cmb_language.currentIndexChanged.connect(self._on_language_changed)

        language_layout.addWidget(self.language_label)
        language_layout.addStretch(1)
        language_layout.addWidget(self.cmb_language)

        status_card = CardWidget(content)
        status_card.setObjectName("statusCard")
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(10)

        self.status_dot = QLabel(status_card)
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setFixedSize(10, 10)

        self.status_label = StrongBodyLabel("Update status: not checked yet", status_card)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        status_layout.addWidget(self.status_dot)
        status_layout.addWidget(self.status_label, 1)

        notes_header = QHBoxLayout()
        notes_header.setContentsMargins(0, 0, 0, 0)
        notes_header.setSpacing(8)
        self.notes_title = SubtitleLabel("Release notes", content)
        self.notes_title.setObjectName("sectionTitle")
        self.notes_hint = BodyLabel("Markdown is rendered with links.", content)
        self.notes_hint.setObjectName("mutedLabel")
        notes_header.addWidget(self.notes_title)
        notes_header.addStretch(1)
        notes_header.addWidget(self.notes_hint)

        self.release_notes = TextBrowser(content)
        self.release_notes.setObjectName("releaseNotes")
        self.release_notes.setReadOnly(True)
        self.release_notes.setOpenExternalLinks(True)
        self.release_notes.setPlaceholderText("Release notes will appear here.")
        self.release_notes.setMinimumHeight(190)

        self.btn_check_updates = PushButton("Check for Updates", content, FIF.SYNC)
        self.btn_check_updates.clicked.connect(self._check_updates)

        self.btn_download_update = PrimaryPushButton("Download Update", content, FIF.DOWNLOAD)
        self.btn_download_update.setObjectName("primaryButton")
        self.btn_download_update.setEnabled(False)
        self.btn_download_update.clicked.connect(self._download_update)

        self.github_link = HyperlinkButton(
            url="https://github.com/Locon213/TranslatorHoi4",
            text="GitHub Releases",
            parent=content,
        )
        self.github_link.setIcon(FIF.GITHUB)

        self.ok_btn = PushButton("Close", content, FIF.CLOSE)
        self.ok_btn.clicked.connect(self.accept)

        for button in (self.btn_check_updates, self.btn_download_update, self.github_link, self.ok_btn):
            button.setMinimumHeight(34)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_check_updates.setMinimumWidth(160)
        self.btn_download_update.setMinimumWidth(165)
        self.github_link.setMinimumWidth(160)
        self.ok_btn.setMinimumWidth(92)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(10)
        buttons.addWidget(self.btn_check_updates)
        buttons.addWidget(self.btn_download_update)
        buttons.addStretch(1)
        buttons.addWidget(self.github_link)
        buttons.addWidget(self.ok_btn)

        content_layout.addWidget(hero)
        content_layout.addWidget(language_card)
        content_layout.addWidget(status_card)
        content_layout.addLayout(notes_header)
        content_layout.addWidget(self.release_notes, 1)
        content_layout.addLayout(buttons)

        layout.addWidget(header)
        layout.addWidget(content, 1)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog#aboutDialog {
                background: #0f1117;
                border: 1px solid #2c3340;
                border-radius: 10px;
            }
            QWidget#aboutContent {
                background: #11141a;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QFrame#titleBar {
                background: #171b22;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                border-bottom: 1px solid #2a313d;
            }
            QLabel#windowTitle {
                color: #dce3ee;
                font-size: 12px;
            }
            QPushButton#titleCloseButton {
                background: transparent;
                border: none;
                color: #dce3ee;
                font-size: 18px;
                font-weight: 500;
                padding: 0;
            }
            QPushButton#titleCloseButton:hover {
                background: #cf3434;
                color: white;
            }
            QLabel#appIcon {
                background: #111821;
                border: 1px solid #394252;
                border-radius: 16px;
            }
            QWidget#heroPanel,
            QWidget#statusCard {
                background: #1b2029;
                border: 1px solid #303846;
                border-radius: 8px;
            }
            QLabel#appTitle {
                color: #f6f8fb;
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#mutedLabel {
                color: #aeb8c6;
            }
            QLabel#sectionTitle {
                color: #f3f6fb;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#statusLabel {
                color: #edf2f8;
                font-weight: 600;
            }
            QTextBrowser#releaseNotes {
                background: #0e1218;
                color: #dce3ee;
                border: 1px solid #303846;
                border-radius: 8px;
                padding: 12px;
                selection-background-color: #139fd6;
            }
            QTextBrowser#releaseNotes:focus {
                border: 1px solid #16c9e6;
            }
            QTextBrowser#releaseNotes QScrollBar:vertical {
                background: #0e1218;
                width: 10px;
                margin: 3px;
            }
            QTextBrowser#releaseNotes QScrollBar::handle:vertical {
                background: #3d4656;
                border-radius: 5px;
                min-height: 28px;
            }
            QTextBrowser#releaseNotes QScrollBar::handle:vertical:hover {
                background: #4d586b;
            }
            QTextBrowser#releaseNotes QScrollBar::add-line:vertical,
            QTextBrowser#releaseNotes QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

    def _t(self, text: str) -> str:
        return translate_text(text, self._lang_code)

    def _set_language_combo(self, lang_code: str) -> None:
        idx = self.cmb_language.findData(lang_code)
        if idx < 0:
            idx = self.cmb_language.findData("english")
        if idx >= 0:
            self.cmb_language.blockSignals(True)
            self.cmb_language.setCurrentIndex(idx)
            self.cmb_language.blockSignals(False)

    def _on_language_changed(self) -> None:
        lang_code = self.cmb_language.currentData()
        if not lang_code or lang_code == self._lang_code:
            return
        self._apply_translations(lang_code)
        self.language_changed.emit(lang_code)

    def _apply_translations(self, lang_code: str) -> None:
        self._lang_code = lang_code or "english"
        self._set_language_combo(self._lang_code)
        self.setWindowTitle(self._t("About TranslatorHoi4"))
        self.window_title.setText(self._t("About TranslatorHoi4"))
        self.btn_title_close.setToolTip(self._t("Close"))
        self.version_label.setText(
            f"{self._t('Version')} {self._version_info['version']} "
            f"({self._version_info['platform']}/{self._version_info['arch']})"
        )
        self.channel_label.setText(f"{self._t('Channel')}: {self._version_info['channel']}")
        self.author_label.setText(self._t("Developed by Locon213"))
        self.language_label.setText(self._t("Interface Language"))
        self.notes_title.setText(self._t("Release notes"))
        self.notes_hint.setText(self._t("Markdown is rendered with links."))
        self.release_notes.setPlaceholderText(self._t("Release notes will appear here."))
        self.btn_check_updates.setText(self._t("Check for Updates"))
        self.btn_download_update.setText(self._t("Download Update"))
        self.github_link.setText(self._t("GitHub Releases"))
        self.ok_btn.setText(self._t("Close"))
        self._render_update_info(self._update_info)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(text)
        self.status_dot.setStyleSheet(
            f"QLabel#statusDot {{ background: {color}; border-radius: 5px; }}"
        )

    def _render_update_info(self, update_info: dict) -> None:
        self._update_info = update_info or {}
        if self._update_info.get("error"):
            self._set_status(f"{self._t('Update status: check failed')} ({self._update_info['error']})", "#f87171")
        elif self._update_info.get("update_available"):
            version = self._update_info.get("latest_version", "unknown")
            asset_name = self._update_info.get("asset_name", "matching asset")
            self._set_status(f"{self._t('Update available')}: {version} ({asset_name})", "#22d3ee")
        elif self._update_info.get("latest_version"):
            self._set_status(f"{self._t('Up to date. Latest version')}: {self._update_info['latest_version']}", "#34d399")
        else:
            self._set_status(self._t("Update status: not checked yet"), "#94a3b8")

        notes = self._update_info.get("release_notes") or self._t("No release notes available.")
        self.release_notes.setMarkdown(notes)
        self.btn_download_update.setEnabled(bool(self._update_info.get("update_available") and self._update_info.get("download_url")))

    def _check_updates(self, force: bool = True) -> None:
        if self._update_thread and self._update_thread.isRunning():
            return
        self._set_status(self._t("Update status: checking..."), "#facc15")
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
        self._set_status(self._t("Update status: downloading update package..."), "#22d3ee")
        self.btn_download_update.setEnabled(False)
        self._download_thread = _UpdateDownloadThread()
        self._download_thread.finished_info.connect(self._on_download_finished)
        self._download_thread.failed.connect(self._on_download_failed)
        self._download_thread.start()

    def _on_download_finished(self, info: dict) -> None:
        version = info.get("latest_version", "unknown")
        message = info.get("message", self._t("Update package prepared."))
        self._set_status(f"{self._t('Update')} {version}: {message}", "#34d399")
        self.btn_download_update.setEnabled(bool(self._update_info.get("download_url")))

    def _on_download_failed(self, error: str) -> None:
        self._set_status(f"{self._t('Update status: failed to download update')} ({error})", "#f87171")
        self.btn_download_update.setEnabled(bool(self._update_info.get("download_url")))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 48:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_position and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_position = None
        super().mouseReleaseEvent(event)
