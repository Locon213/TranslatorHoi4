"""Main application window using PyQt6-Fluent-Widgets."""
from __future__ import annotations

import json
import os
from typing import Optional

from PyQt6.QtCore import Qt, QSize, QUrl, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QAction, QDesktopServices, QIcon, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QScrollArea, QSizePolicy
)

# --- Fluent Widgets Imports ---
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, PushButton, PrimaryPushButton,
    LineEdit, ComboBox, SpinBox, CheckBox, SwitchButton,
    FluentIcon as FIF, setTheme, Theme,
    InfoBar, InfoBarPosition, ProgressBar, TextEdit,
    CardWidget, SimpleCardWidget, HeaderCardWidget,
    StrongBodyLabel, BodyLabel, SubtitleLabel,
    ScrollArea, MessageBox
)

# --- Project Imports ---
from ..parsers.paradox_yaml import LANG_NAME_LIST, parse_yaml_file
from ..utils.fs import collect_localisation_files
from ..translator.engine import JobConfig, MODEL_REGISTRY, TranslateWorker, TestModelWorker
from .about import AboutDialog
from .review_window import ReviewInterface

# --- Custom Widgets for Styling ---

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
    """Simple spinning arc indicator (Kept from original)."""
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        pen = QPen(QColor(255, 255, 255) if Theme.DARK else QColor(0, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawArc(-6, -6, 12, 12, 0, 270 * 16)

class IOModelFetchThread(QThread):
    ready = pyqtSignal(list)
    fail = pyqtSignal(str)

    def __init__(self, api_key: Optional[str], base_url: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url

    def run(self):
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = requests.get(self.base_url.rstrip('/') + "/models", headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
            self.ready.emit(models)
        except Exception as e:
            self.fail.emit(str(e))


# --- Interface Pages ---

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


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        
        # 1. Setup Window
        self.setWindowTitle("HOI4 Localizer ✨")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.resize(1100, 750)
        self._total_files = 0
        
        # 2. Theme
        setTheme(Theme.DARK)

        # 3. Initialize Variables & Workers
        self._worker: Optional[TranslateWorker] = None
        self._test_thread: Optional[TestModelWorker] = None
        self._io_fetch_thread: Optional[IOModelFetchThread] = None

        # 4. Create UI Components
        self._init_components()

        # 5. Build Interfaces
        self._init_navigation()

        # 6. Apply logic hooks
        self._switch_backend_settings(self.cmb_model.currentText())
        

    def _init_components(self):
        """Initialize all input widgets here so 'self.variable' works globally."""
        
        # --- BASIC ---
        self.ed_src = LineEdit()
        self.ed_src.setPlaceholderText("Path to mod folder")
        self.ed_out = LineEdit()
        self.ed_out.setPlaceholderText("Output folder (leave empty for in-place)")
        self.ed_prev = LineEdit()
        self.ed_prev.setPlaceholderText("Optional: previous translation folder")
        
        self.chk_inplace = CheckBox("Translate in-place (overwrite)")
        self.chk_inplace.stateChanged.connect(self._toggle_inplace)
        
        self.cmb_src_lang = ComboBox()
        self.cmb_src_lang.addItems(LANG_NAME_LIST)
        self.cmb_src_lang.setCurrentText("english")
        
        self.cmb_dst_lang = ComboBox()
        self.cmb_dst_lang.addItems(LANG_NAME_LIST)
        self.cmb_dst_lang.setCurrentText("russian")
        
        self.cmb_model = ComboBox()
        self.cmb_model.addItems(list(MODEL_REGISTRY.keys()))
        self.cmb_model.setCurrentText("G4F: API (g4f.dev)")
        self.cmb_model.currentTextChanged.connect(self._switch_backend_settings)

        self.spn_temp = SpinBox()
        self.spn_temp.setRange(1, 120)
        self.spn_temp.setValue(70)

        self.chk_skip_exist = CheckBox("Skip existing files")
        self.chk_skip_exist.setChecked(False)
        self.chk_mark_loc = CheckBox("Mark translated lines (#LOC!)")
        self.chk_mark_loc.setChecked(True)
        self.chk_reuse_prev = CheckBox("Reuse previous translations")
        self.chk_reuse_prev.setChecked(True)

        # Action Buttons
        self.btn_scan = PushButton("Scan Files", self, FIF.SEARCH)
        self.btn_scan.clicked.connect(self._scan_files)
        
        self.btn_test = PushButton("Test Connection", self, FIF.LINK)
        self.btn_test.clicked.connect(self._test_model_async)
        
        self.btn_go = PrimaryPushButton("Start Translating", self, FIF.PLAY)
        self.btn_go.clicked.connect(self._start)
        
        self.btn_cancel = PushButton("Cancel", self, FIF.CANCEL)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)

        # --- ADVANCED ---
        self.chk_strip_md = CheckBox("Strip Markdown")
        self.chk_strip_md.setChecked(True)
        self.chk_rename_files = CheckBox("Auto-rename files (*_l_russian.yml)")
        self.chk_rename_files.setChecked(True)
        
        self.ed_key_skip = LineEdit()
        self.ed_key_skip.setPlaceholderText("Regex: ^STATE_")

        self.spn_batch = SpinBox(); self.spn_batch.setRange(1, 200); self.spn_batch.setValue(12)
        self.spn_files_cc = SpinBox(); self.spn_files_cc.setRange(1, 6); self.spn_files_cc.setValue(1)

        # G4F (Updated)
        self.ed_g4f_model = LineEdit(); self.ed_g4f_model.setText("gpt-4o")
        self.ed_g4f_api_key = LineEdit(); self.ed_g4f_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.chk_g4f_async = CheckBox("Use Async"); self.chk_g4f_async.setChecked(True)
        self.spn_g4f_cc = SpinBox(); self.spn_g4f_cc.setRange(1, 50); self.spn_g4f_cc.setValue(6)
        
        self.btn_g4f_key = PushButton("Get API Key", self, FIF.LINK)
        self.btn_g4f_key.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://g4f.dev/api_key.html")))
        
        self.btn_g4f_models = PushButton("View Models List", self, FIF.SEARCH)
        self.btn_g4f_models.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://g4f.dev/v1/models")))

        # IO
        self.ed_io_api_key = LineEdit(); self.ed_io_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_io_base = LineEdit()
        self.cmb_io_model = ComboBox(); self.cmb_io_model.setEnabled(False)
        self.io_loader = LoadingIndicator()
        self.chk_io_async = CheckBox("Use Async"); self.chk_io_async.setChecked(True)
        self.spn_io_cc = SpinBox(); self.spn_io_cc.setValue(6)

        # OpenAI
        self.ed_openai_api_key = LineEdit(); self.ed_openai_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_openai_base = LineEdit()
        self.ed_openai_model = LineEdit(); self.ed_openai_model.setPlaceholderText("gpt-4")
        self.chk_openai_async = CheckBox("Use Async"); self.chk_openai_async.setChecked(True)
        self.spn_openai_cc = SpinBox(); self.spn_openai_cc.setValue(6)

        # Anthropic
        self.ed_anthropic_api_key = LineEdit(); self.ed_anthropic_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_anthropic_model = LineEdit(); self.ed_anthropic_model.setPlaceholderText("claude-sonnet-4-5-20250929")
        self.chk_anthropic_async = CheckBox("Use Async"); self.chk_anthropic_async.setChecked(True)
        self.spn_anthropic_cc = SpinBox(); self.spn_anthropic_cc.setValue(6)

        # Gemini
        self.ed_gemini_api_key = LineEdit(); self.ed_gemini_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_gemini_model = LineEdit(); self.ed_gemini_model.setPlaceholderText("gemini-2.5-flash")
        self.chk_gemini_async = CheckBox("Use Async"); self.chk_gemini_async.setChecked(True)
        self.spn_gemini_cc = SpinBox(); self.spn_gemini_cc.setValue(6)

        # --- TOOLS ---
        self.ed_glossary = LineEdit()
        self.ed_cache = LineEdit()

        # --- MONITOR (Logs) ---
        self.pb_global = ProgressBar()
        self.pb_file = ProgressBar()
        self.lbl_stats = BodyLabel("Words: 0 | Keys: 0 | Files: 0/0")
        self.lbl_file = BodyLabel("Ready")
        self.txt_log = TextEdit()
        self.txt_log.setReadOnly(True)

    def _init_navigation(self):
        # 1. HOME Interface
        self.home_interface = BaseInterface("HomeInterface", self)
        
        # Section: Paths
        self.home_interface.vBoxLayout.addWidget(SectionHeader("Mod Paths"))
        
        card_src = CardWidget(self.home_interface)
        l_src = QVBoxLayout(card_src)
        h_src = QHBoxLayout()
        btn_src_browse = PushButton("Browse")
        btn_src_browse.clicked.connect(self._pick_src)
        h_src.addWidget(self.ed_src, 1); h_src.addWidget(btn_src_browse)
        l_src.addWidget(BodyLabel("Source Mod Folder:"))
        l_src.addLayout(h_src)
        self.home_interface.vBoxLayout.addWidget(card_src)

        card_out = CardWidget(self.home_interface)
        l_out = QVBoxLayout(card_out)
        h_out = QHBoxLayout()
        btn_out_browse = PushButton("Browse")
        btn_out_browse.clicked.connect(self._pick_out)
        h_out.addWidget(self.ed_out, 1); h_out.addWidget(btn_out_browse)
        l_out.addWidget(BodyLabel("Output Folder:"))
        l_out.addLayout(h_out)
        l_out.addWidget(self.chk_inplace)
        l_out.addWidget(self.chk_skip_exist)
        self.home_interface.vBoxLayout.addWidget(card_out)

        # Section: Settings
        self.home_interface.vBoxLayout.addWidget(SectionHeader("General Settings"))
        
        row_lang = QHBoxLayout()
        row_lang.addWidget(SettingCard("Source Language", self.cmb_src_lang))
        row_lang.addWidget(SettingCard("Target Language", self.cmb_dst_lang))
        self.home_interface.vBoxLayout.addLayout(row_lang)

        self.home_interface.vBoxLayout.addWidget(SettingCard("AI Model", self.cmb_model))
        
        row_params = QHBoxLayout()
        row_params.addWidget(SettingCard("Temperature x100", self.spn_temp))
        row_params.addWidget(SettingCard("Reuse #LOC!", self.chk_reuse_prev))
        self.home_interface.vBoxLayout.addLayout(row_params)

        # Section: Actions
        self.home_interface.vBoxLayout.addStretch(1)
        action_bar = CardWidget(self.home_interface)
        l_act = QHBoxLayout(action_bar)
        l_act.addWidget(self.btn_scan)
        l_act.addStretch(1)
        l_act.addWidget(self.btn_test)
        l_act.addWidget(self.btn_cancel)
        l_act.addWidget(self.btn_go)
        self.home_interface.vBoxLayout.addWidget(action_bar)

        # 2. ADVANCED Interface
        self.adv_interface = BaseInterface("AdvancedInterface", self)
        
        # Processing
        self.adv_interface.vBoxLayout.addWidget(SectionHeader("Processing Rules"))
        proc_card = CardWidget()
        proc_l = QVBoxLayout(proc_card)
        proc_l.addWidget(self.chk_strip_md)
        proc_l.addWidget(self.chk_rename_files)
        h_skip = QHBoxLayout(); h_skip.addWidget(BodyLabel("Skip Regex:")); h_skip.addWidget(self.ed_key_skip)
        proc_l.addLayout(h_skip)
        self.adv_interface.vBoxLayout.addWidget(proc_card)

        # Model Specific Containers
        self.adv_interface.vBoxLayout.addWidget(SectionHeader("Model Specific Settings"))

        # G4F
        self.g4f_container = CardWidget()
        l = QVBoxLayout(self.g4f_container)
        l.addWidget(StrongBodyLabel("G4F API Settings (g4f.dev)"))
        l.addWidget(SettingCard("Model Name", self.ed_g4f_model))
        l.addWidget(SettingCard("API Key", self.ed_g4f_api_key))
        
        h_g4f_btns = QHBoxLayout()
        h_g4f_btns.addWidget(self.btn_g4f_key)
        h_g4f_btns.addWidget(self.btn_g4f_models)
        l.addLayout(h_g4f_btns)

        l.addWidget(self.chk_g4f_async)
        l.addWidget(SettingCard("Concurrency", self.spn_g4f_cc))
        self.adv_interface.vBoxLayout.addWidget(self.g4f_container)

        # IO
        self.io_container = CardWidget()
        l = QVBoxLayout(self.io_container)
        l.addWidget(StrongBodyLabel("IO Intelligence"))
        l.addWidget(SettingCard("API Key", self.ed_io_api_key))
        h_io = QHBoxLayout(); h_io.addWidget(self.cmb_io_model); h_io.addWidget(self.io_loader)
        l.addLayout(h_io)
        self.adv_interface.vBoxLayout.addWidget(self.io_container)

        # OpenAI
        self.openai_container = CardWidget()
        l = QVBoxLayout(self.openai_container)
        l.addWidget(StrongBodyLabel("OpenAI Compatible"))
        l.addWidget(SettingCard("Base URL", self.ed_openai_base))
        l.addWidget(SettingCard("API Key", self.ed_openai_api_key))
        l.addWidget(SettingCard("Model ID", self.ed_openai_model))
        self.adv_interface.vBoxLayout.addWidget(self.openai_container)

        # Anthropic
        self.anthropic_container = CardWidget()
        l = QVBoxLayout(self.anthropic_container)
        l.addWidget(StrongBodyLabel("Anthropic (Claude)"))
        l.addWidget(SettingCard("API Key", self.ed_anthropic_api_key))
        l.addWidget(SettingCard("Model", self.ed_anthropic_model))
        self.adv_interface.vBoxLayout.addWidget(self.anthropic_container)

        # Gemini
        self.gemini_container = CardWidget()
        l = QVBoxLayout(self.gemini_container)
        l.addWidget(StrongBodyLabel("Google Gemini"))
        l.addWidget(SettingCard("API Key", self.ed_gemini_api_key))
        l.addWidget(SettingCard("Model", self.ed_gemini_model))
        self.adv_interface.vBoxLayout.addWidget(self.gemini_container)

        # 3. TOOLS Interface
        self.tools_interface = BaseInterface("ToolsInterface", self)
        self.tools_interface.vBoxLayout.addWidget(SectionHeader("Data Management"))
        
        btn_gloss = PushButton("Load CSV"); btn_gloss.clicked.connect(self._pick_glossary)
        self.tools_interface.vBoxLayout.addWidget(SettingCard("Glossary Path", self.ed_glossary))
        
        btn_clear = PushButton("Clear Cache"); btn_clear.clicked.connect(self._clear_cache)
        self.tools_interface.vBoxLayout.addWidget(SettingCard("Cache File", self.ed_cache))
        self.tools_interface.vBoxLayout.addWidget(btn_clear)

        self.tools_interface.vBoxLayout.addWidget(SectionHeader("Presets"))
        h_preset = QHBoxLayout()
        btn_save = PushButton("Save Preset", self, FIF.SAVE)
        btn_save.clicked.connect(self._save_preset)
        btn_load = PushButton("Load Preset", self, FIF.FOLDER)
        btn_load.clicked.connect(self._load_preset)
        h_preset.addWidget(btn_save); h_preset.addWidget(btn_load)
        self.tools_interface.vBoxLayout.addLayout(h_preset)

        # 4. MONITOR Interface
        self.monitor_interface = BaseInterface("MonitorInterface", self)
        
        card_prog = CardWidget(self.monitor_interface)
        l_prog = QVBoxLayout(card_prog)
        l_prog.addWidget(StrongBodyLabel("Total Progress"))
        l_prog.addWidget(self.pb_global)
        l_prog.addWidget(self.lbl_stats)
        l_prog.addSpacing(10)
        l_prog.addWidget(StrongBodyLabel("Current File"))
        l_prog.addWidget(self.lbl_file)
        l_prog.addWidget(self.pb_file)
        self.monitor_interface.vBoxLayout.addWidget(card_prog)

        self.monitor_interface.vBoxLayout.addWidget(SectionHeader("Application Log"))
        self.monitor_interface.vBoxLayout.addWidget(self.txt_log)


        # --- ADD TO WINDOW ---
        self.addSubInterface(self.home_interface, FIF.HOME, "Home")
        self.addSubInterface(self.adv_interface, FIF.SETTING, "Advanced Settings")
        self.addSubInterface(self.tools_interface, FIF.DEVELOPER_TOOLS, "Tools")
        self.addSubInterface(self.monitor_interface, FIF.COMMAND_PROMPT, "Process Monitor")
        
        # Review Interface
        self.review_interface = ReviewInterface(self)
        self.addSubInterface(self.review_interface, FIF.EDIT, "Review & Edit")
        
        # About button at bottom
        self.navigationInterface.addItem(
            routeKey="About",
            icon=FIF.INFO,
            text="About",
            onClick=self._show_about,
            selectable=False,
            position=NavigationItemPosition.BOTTOM
        )

    # --- LOGIC METHODS ---

    def _show_about(self):
        title = "HoI4 Translator"
        content = "Version 1.2\nUsing PyQt6-Fluent-Widgets"
        w = MessageBox(title, content, self)
        w.exec()

    def _switch_backend_settings(self, text: str):
        # Logic to hide/show specific cards in Advanced tab
        self.g4f_container.setVisible(text == "G4F: API (g4f.dev)")
        self.io_container.setVisible(text == "IO: chat.completions")
        self.openai_container.setVisible(text == "OpenAI Compatible API")
        self.anthropic_container.setVisible(text == "Anthropic: Claude")
        self.gemini_container.setVisible(text == "Google: Gemini")

        if text == "IO: chat.completions":
            self._refresh_io_models()

    def _refresh_io_models(self):
        if self._io_fetch_thread is not None:
            self._io_fetch_thread.quit()
            self._io_fetch_thread.wait()
        self.cmb_io_model.clear()
        self.cmb_io_model.addItem("Please wait…")
        self.cmb_io_model.setEnabled(False)
        self.io_loader.start()
        api_key = self.ed_io_api_key.text().strip() or None
        base = self.ed_io_base.text().strip() or "https://api.intelligence.io.solutions/api/v1/"
        self._io_fetch_thread = IOModelFetchThread(api_key, base)
        self._io_fetch_thread.ready.connect(self._on_io_models_ready)
        self._io_fetch_thread.fail.connect(self._on_io_models_fail)
        self._io_fetch_thread.start()

    def _on_io_models_ready(self, models: list):
        self.io_loader.stop()
        self.cmb_io_model.setEnabled(True)
        self.cmb_io_model.clear()
        if models:
            self.cmb_io_model.addItems(models)
        else:
            self.cmb_io_model.addItem("No models")

    def _on_io_models_fail(self, err: str):
        self.io_loader.stop()
        self.cmb_io_model.setEnabled(False)
        self.cmb_io_model.clear()
        self.cmb_io_model.addItem("Failed to load")

    def _pick_src(self):
        d = QFileDialog.getExistingDirectory(self, "Select source mod folder")
        if d: self.ed_src.setText(d)

    def _pick_prev(self):
        d = QFileDialog.getExistingDirectory(self, "Select previous localized folder")
        if d: self.ed_prev.setText(d)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self.ed_out.setText(d)

    def _open_out(self):
        p = self.ed_out.text().strip() or self.ed_src.text().strip()
        if p and os.path.isdir(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _toggle_inplace(self):
        v = self.chk_inplace.isChecked()
        self.ed_out.setDisabled(v)

    def _scan_files(self):
        src = self.ed_src.text().strip()
        if not src or not os.path.isdir(src):
            InfoBar.warning(title="Scan Error", content="Please choose a valid source folder", parent=self)
            return
        files = collect_localisation_files(src)
        self._total_files = len(files)
        self._append_log(f"Found {len(files)} localisation files.")
        self.lbl_stats.setText(f"Words: 0 | Keys: 0 | Files: 0/{self._total_files}")
        
        # Switch to monitor tab to show result
        self.switchTo(self.monitor_interface)

    def _pick_glossary(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select glossary CSV", "", "CSV files (*.csv)")
        if p:
            self.ed_glossary.setText(p)

    def _clear_cache(self):
        p = self.ed_cache.text().strip()
        if not p:
            out = self.ed_out.text().strip() or self.ed_src.text().strip()
            p = os.path.join(out, ".hoi4loc_cache.json") if out else ""
        if p and os.path.isfile(p):
            try:
                os.remove(p)
                self._append_log(f"Cache cleared: {p}")
                InfoBar.success("Success", "Cache cleared", parent=self)
            except Exception as e:
                self._append_log(f"Failed to clear cache: {e}")
        else:
            InfoBar.info("Info", "No cache file found", parent=self)

    def _save_preset(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save preset", "", "JSON (*.json)")
        if not p: return
        data = {
            "src": self.ed_src.text(),
            "out": self.ed_out.text(),
            "prev": self.ed_prev.text(),
            "in_place": self.chk_inplace.isChecked(),
            "src_lang": self.cmb_src_lang.currentText(),
            "dst_lang": self.cmb_dst_lang.currentText(),
            "model": self.cmb_model.currentText(),
            "temp_x100": self.spn_temp.value(),
            "skip_existing": self.chk_skip_exist.isChecked(),
            "strip_md": self.chk_strip_md.isChecked(),
            "rename_files": self.chk_rename_files.isChecked(),
            "key_skip_regex": self.ed_key_skip.text(),
            "batch_size": self.spn_batch.value(),
            "files_cc": self.spn_files_cc.value(),
            "glossary": self.ed_glossary.text(),
            "cache": self.ed_cache.text(),
            "reuse_prev_loc": self.chk_reuse_prev.isChecked(),
            "mark_loc": self.chk_mark_loc.isChecked(),
            "g4f_model": self.ed_g4f_model.text(),
            "g4f_api_key": self.ed_g4f_api_key.text(),
            "g4f_async": self.chk_g4f_async.isChecked(),
            "g4f_cc": self.spn_g4f_cc.value(),
            "io_model": self.cmb_io_model.currentText(),
            "io_api_key": self.ed_io_api_key.text(),
            "io_base_url": self.ed_io_base.text(),
            "io_async": self.chk_io_async.isChecked(),
            "io_cc": self.spn_io_cc.value(),
            "openai_api_key": self.ed_openai_api_key.text(),
            "openai_base_url": self.ed_openai_base.text(),
            "openai_model": self.ed_openai_model.text(),
            "openai_async": self.chk_openai_async.isChecked(),
            "openai_cc": self.spn_openai_cc.value(),
            "anthropic_api_key": self.ed_anthropic_api_key.text(),
            "anthropic_model": self.ed_anthropic_model.text(),
            "anthropic_async": self.chk_anthropic_async.isChecked(),
            "anthropic_cc": self.spn_anthropic_cc.value(),
            "gemini_api_key": self.ed_gemini_api_key.text(),
            "gemini_model": self.ed_gemini_model.text(),
            "gemini_async": self.chk_gemini_async.isChecked(),
            "gemini_cc": self.spn_gemini_cc.value(),
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._append_log(f"Preset saved → {p}")
            InfoBar.success("Preset Saved", f"Saved to {os.path.basename(p)}", parent=self)
        except Exception as e:
            InfoBar.error("Error", f"Failed to save: {e}", parent=self)

    def _load_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if not p: return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ed_src.setText(data.get("src",""))
            self.ed_out.setText(data.get("out",""))
            self.ed_prev.setText(data.get("prev",""))
            self.chk_inplace.setChecked(bool(data.get("in_place", False)))
            self.cmb_src_lang.setCurrentText(data.get("src_lang","english"))
            self.cmb_dst_lang.setCurrentText(data.get("dst_lang","russian"))
            model = data.get("model", "G4F: API (g4f.dev)")
            if model in MODEL_REGISTRY: self.cmb_model.setCurrentText(model)
            self.spn_temp.setValue(int(data.get("temp_x100", 70)))
            self.chk_skip_exist.setChecked(bool(data.get("skip_existing", False)))
            self.chk_strip_md.setChecked(bool(data.get("strip_md", True)))
            self.chk_rename_files.setChecked(bool(data.get("rename_files", True)))
            self.ed_key_skip.setText(data.get("key_skip_regex",""))
            self.spn_batch.setValue(int(data.get("batch_size", 12)))
            self.spn_files_cc.setValue(int(data.get("files_cc", 1)))
            self.ed_glossary.setText(data.get("glossary",""))
            self.ed_cache.setText(data.get("cache",""))
            self.chk_reuse_prev.setChecked(bool(data.get("reuse_prev_loc", True)))
            self.chk_mark_loc.setChecked(bool(data.get("mark_loc", True)))
            self.ed_g4f_model.setText(data.get("g4f_model","gpt-4o"))
            self.ed_g4f_api_key.setText(data.get("g4f_api_key",""))
            self.chk_g4f_async.setChecked(bool(data.get("g4f_async", True)))
            self.spn_g4f_cc.setValue(int(data.get("g4f_cc", 6)))
            self.ed_io_api_key.setText(data.get("io_api_key",""))
            self.ed_io_base.setText(data.get("io_base_url",""))
            self.chk_io_async.setChecked(bool(data.get("io_async", True)))
            self.spn_io_cc.setValue(int(data.get("io_cc", 6)))
            io_model = data.get("io_model")
            if io_model:
                self.cmb_io_model.clear(); self.cmb_io_model.addItem(io_model); self.cmb_io_model.setCurrentText(io_model)
            self.ed_openai_api_key.setText(data.get("openai_api_key",""))
            self.ed_openai_base.setText(data.get("openai_base_url",""))
            self.ed_openai_model.setText(data.get("openai_model",""))
            self.chk_openai_async.setChecked(bool(data.get("openai_async", True)))
            self.spn_openai_cc.setValue(int(data.get("openai_cc", 6)))
            self.ed_anthropic_api_key.setText(data.get("anthropic_api_key",""))
            self.ed_anthropic_model.setText(data.get("anthropic_model",""))
            self.chk_anthropic_async.setChecked(bool(data.get("anthropic_async", True)))
            self.spn_anthropic_cc.setValue(int(data.get("anthropic_cc", 6)))
            self.ed_gemini_api_key.setText(data.get("gemini_api_key",""))
            self.ed_gemini_model.setText(data.get("gemini_model",""))
            self.chk_gemini_async.setChecked(bool(data.get("gemini_async", True)))
            self.spn_gemini_cc.setValue(int(data.get("gemini_cc", 6)))

            self._append_log(f"Preset loaded ← {p}")
            InfoBar.success("Preset Loaded", "Settings restored", parent=self)
        except Exception as e:
            InfoBar.error("Error", f"Failed to load: {e}", parent=self)

    def _test_model_async(self):
        if self._test_thread is not None:
            InfoBar.warning("Test", "Test is already running.", parent=self); return
        self.btn_test.setEnabled(False)
        self.switchTo(self.monitor_interface)
        self._append_log("Starting connection test...")
        
        self._test_thread = TestModelWorker(
            model_key=self.cmb_model.currentText(),
            src_lang=self.cmb_src_lang.currentText(),
            dst_lang=self.cmb_dst_lang.currentText(),
            temperature=self.spn_temp.value() / 100.0,
            strip_md=self.chk_strip_md.isChecked(),
            glossary_path=self.ed_glossary.text().strip() or None,
            g4f_model=self.ed_g4f_model.text().strip() or None,
            g4f_api_key=self.ed_g4f_api_key.text().strip() or None,
            g4f_async=self.chk_g4f_async.isChecked(),
            g4f_concurrency=self.spn_g4f_cc.value(),
            io_model=self.cmb_io_model.currentText().strip() or None,
            io_api_key=self.ed_io_api_key.text().strip() or None,
            io_base_url=self.ed_io_base.text().strip() or None,
            io_async=self.chk_io_async.isChecked(),
            io_concurrency=self.spn_io_cc.value(),
            openai_api_key=self.ed_openai_api_key.text().strip() or None,
            openai_model=self.ed_openai_model.text().strip() or None,
            openai_base_url=self.ed_openai_base.text().strip() or None,
            openai_async=self.chk_openai_async.isChecked(),
            openai_concurrency=self.spn_openai_cc.value(),
            anthropic_api_key=self.ed_anthropic_api_key.text().strip() or None,
            anthropic_model=self.ed_anthropic_model.text().strip() or "claude-sonnet-4-5-20250929",
            anthropic_async=self.chk_anthropic_async.isChecked(),
            anthropic_concurrency=self.spn_anthropic_cc.value(),
            gemini_api_key=self.ed_gemini_api_key.text().strip() or None,
            gemini_model=self.ed_gemini_model.text().strip() or "gemini-2.5-flash",
            gemini_async=self.chk_gemini_async.isChecked(),
            gemini_concurrency=self.spn_gemini_cc.value(),
        )
        self._test_thread.ok.connect(self._on_test_ok)
        self._test_thread.fail.connect(self._on_test_fail)
        self._test_thread.finished.connect(self._on_test_finished, Qt.ConnectionType.QueuedConnection)
        self._test_thread.start()

    def _on_test_ok(self, txt: str):
        self._append_log(f"Test OK → {txt}")
        InfoBar.success("Test OK", txt, parent=self)

    def _on_test_fail(self, e: str):
        self._append_log(f"Test failed: {e}")
        InfoBar.error("Test Failed", str(e), parent=self)

    def _on_test_finished(self):
        self.btn_test.setEnabled(True)
        self._test_thread = None

    def _start(self):
        # Auto-switch to Monitor tab
        self.switchTo(self.monitor_interface)
        
        src = self.ed_src.text().strip()
        if not src or not os.path.isdir(src):
            InfoBar.warning("Start", "Please choose a valid source folder", parent=self); return
        in_place = self.chk_inplace.isChecked()
        out = self.ed_out.text().strip()
        if not in_place and (not out or not os.path.isdir(out)):
            InfoBar.warning("Start", "Please choose a valid output folder", parent=self); return
        
        cache_path = (self.ed_cache.text().strip() or None)
        if cache_path is None:
            base = out or src
            cache_path = os.path.join(base, ".hoi4loc_cache.json")
        cfg = JobConfig(
            src_dir=src,
            out_dir=out or src,
            src_lang=self.cmb_src_lang.currentText(),
            dst_lang=self.cmb_dst_lang.currentText(),
            model_key=self.cmb_model.currentText(),
            temperature=self.spn_temp.value() / 100.0,
            in_place=in_place,
            skip_existing=self.chk_skip_exist.isChecked(),
            strip_md=self.chk_strip_md.isChecked(),
            batch_size=self.spn_batch.value(),
            rename_files=self.chk_rename_files.isChecked(),
            files_concurrency=self.spn_files_cc.value(),
            key_skip_regex=(self.ed_key_skip.text().strip() or None),
            cache_path=cache_path,
            glossary_path=(self.ed_glossary.text().strip() or None),
            prev_loc_dir=(self.ed_prev.text().strip() or None),
            reuse_prev_loc=self.chk_reuse_prev.isChecked(),
            mark_loc_flag=self.chk_mark_loc.isChecked(),
            g4f_model=self.ed_g4f_model.text().strip() or "gpt-4o",
            g4f_api_key=self.ed_g4f_api_key.text().strip() or None,
            g4f_async=self.chk_g4f_async.isChecked(),
            g4f_concurrency=self.spn_g4f_cc.value(),
            io_model=self.cmb_io_model.currentText().strip() or "meta-llama/Llama-3.3-70B-Instruct",
            io_api_key=self.ed_io_api_key.text().strip() or None,
            io_base_url=self.ed_io_base.text().strip() or None,
            io_async=self.chk_io_async.isChecked(),
            io_concurrency=self.spn_io_cc.value(),
            openai_api_key=self.ed_openai_api_key.text().strip() or None,
            openai_model=self.ed_openai_model.text().strip() or "gpt-4",
            openai_base_url=self.ed_openai_base.text().strip() or None,
            openai_async=self.chk_openai_async.isChecked(),
            openai_concurrency=self.spn_openai_cc.value(),
            anthropic_api_key=self.ed_anthropic_api_key.text().strip() or None,
            anthropic_model=self.ed_anthropic_model.text().strip() or "claude-sonnet-4-5-20250929",
            anthropic_async=self.chk_anthropic_async.isChecked(),
            anthropic_concurrency=self.spn_anthropic_cc.value(),
            gemini_api_key=self.ed_gemini_api_key.text().strip() or None,
            gemini_model=self.ed_gemini_model.text().strip() or "gemini-2.5-flash",
            gemini_async=self.chk_gemini_async.isChecked(),
            gemini_concurrency=self.spn_gemini_cc.value(),
        )

        if cfg.model_key == "G4F: API (g4f.dev)":
            os.environ["G4F_MODEL"] = (cfg.g4f_model or "gpt-4o")
            os.environ["G4F_API_KEY"] = (cfg.g4f_api_key or "")
            os.environ["G4F_TEMP"] = str(cfg.temperature)
            os.environ["G4F_ASYNC"] = "1" if cfg.g4f_async else "0"
            os.environ["G4F_CONCURRENCY"] = str(cfg.g4f_concurrency)
        elif cfg.model_key == "IO: chat.completions":
            os.environ["IO_MODEL"] = (cfg.io_model or "meta-llama/Llama-3.3-70B-Instruct")
            os.environ["IO_API_KEY"] = (cfg.io_api_key or "")
            os.environ["IO_BASE_URL"] = (cfg.io_base_url or "https://api.intelligence.io.solutions/api/v1/")
            os.environ["IO_TEMP"] = str(cfg.temperature)
            os.environ["IO_ASYNC"] = "1" if cfg.io_async else "0"
            os.environ["IO_CONCURRENCY"] = str(cfg.io_concurrency)
        elif cfg.model_key == "OpenAI Compatible API":
            os.environ["OPENAI_MODEL"] = (cfg.openai_model or "gpt-4")
            os.environ["OPENAI_API_KEY"] = (cfg.openai_api_key or "")
            os.environ["OPENAI_BASE_URL"] = (cfg.openai_base_url or "https://api.openai.com/v1/")
            os.environ["OPENAI_TEMP"] = str(cfg.temperature)
            os.environ["OPENAI_ASYNC"] = "1" if cfg.openai_async else "0"
            os.environ["OPENAI_CONCURRENCY"] = str(cfg.openai_concurrency)
        elif cfg.model_key == "Anthropic: Claude":
            os.environ["ANTHROPIC_MODEL"] = (cfg.anthropic_model or "claude-sonnet-4-5-20250929")
            os.environ["ANTHROPIC_API_KEY"] = (cfg.anthropic_api_key or "")
            os.environ["ANTHROPIC_TEMP"] = str(cfg.temperature)
            os.environ["ANTHROPIC_ASYNC"] = "1" if cfg.anthropic_async else "0"
            os.environ["ANTHROPIC_CONCURRENCY"] = str(cfg.anthropic_concurrency)
        elif cfg.model_key == "Google: Gemini":
            os.environ["GEMINI_MODEL"] = (cfg.gemini_model or "gemini-2.5-flash")
            os.environ["GEMINI_API_KEY"] = (cfg.gemini_api_key or "")
            os.environ["GEMINI_TEMP"] = str(cfg.temperature)
            os.environ["GEMINI_ASYNC"] = "1" if cfg.gemini_async else "0"
            os.environ["GEMINI_CONCURRENCY"] = str(cfg.gemini_concurrency)

        self._append_log(
            f"Starting with {cfg.model_key} (temp={cfg.temperature}, files_cc={cfg.files_concurrency})…"
        )
        self.btn_go.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.pb_global.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_file.setText("Preparing...")
        self._total_files = len(collect_localisation_files(cfg.src_dir))
        self.lbl_stats.setText(f"Words: 0 | Keys: 0 | Files: 0/{self._total_files}")
        self._worker = TranslateWorker(cfg)
        self._worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.file_progress.connect(self._on_file, Qt.ConnectionType.QueuedConnection)
        self._worker.file_inner_progress.connect(self._on_file_inner, Qt.ConnectionType.QueuedConnection)
        self._worker.log.connect(self._append_log, Qt.ConnectionType.QueuedConnection)
        self._worker.stats.connect(self._on_stats, Qt.ConnectionType.QueuedConnection)
        self._worker.finished_ok.connect(self._on_done, Qt.ConnectionType.QueuedConnection)
        self._worker.aborted.connect(self._on_aborted, Qt.ConnectionType.QueuedConnection)
        self._worker.start()
        
        self.review_interface.save_requested.connect(self._on_review_save)
        self.review_interface.retranslate_requested.connect(self._on_review_retranslate)

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._append_log("Cancelling…")

    def _on_progress(self, cur: int, total: int):
        self.pb_global.setValue(int((cur / max(1, total)) * 100))

    def _on_file(self, relpath: str):
        self.lbl_file.setText(relpath)
        self.pb_file.setValue(0)

    def _on_file_inner(self, cur: int, total: int):
        self.pb_file.setValue(int((cur / max(1, total)) * 100))

    def _on_stats(self, words: int, keys: int, files_done: int):
        self.lbl_stats.setText(f"Words: {words} | Keys: {keys} | Files: {files_done}/{self._total_files}")

    def _on_done(self):
        self._append_log("All done! ✨")
        InfoBar.success("Finished", "Translation completed successfully", parent=self)
        self.btn_go.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_file.setText("Completed")
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(2000)
        except Exception:
            pass
        self._worker = None
        
        self.switchTo(self.review_interface)
        self._load_review_data()

    def _on_aborted(self, msg: str):
        self._append_log(f"Aborted: {msg}")
        InfoBar.error("Aborted", msg, parent=self)
        self.btn_go.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(2000)
        except Exception:
            pass
        self._worker = None

    def _append_log(self, s: str):
        self.txt_log.append(s)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def _load_review_data(self):
        try:
            out_dir = self.ed_out.text().strip()
            if not out_dir:
                out_dir = self.ed_src.text().strip()
            
            if not out_dir:
                return
                
            files = collect_localisation_files(out_dir)
            if not files:
                self._append_log("No localisation files found for review")
                return
                
            file_path = files[0]
            self._append_log(f"Loading {file_path} for review")
            
            data = parse_yaml_file(file_path)
            
            if data:
                self.review_interface.load_data(file_path, data)
                self._append_log(f"Loaded {len(data)} entries for review")
            else:
                self._append_log("No data found in the file")
                
        except Exception as e:
            self._append_log(f"Error loading review data: {e}")

    def _on_review_save(self, data: list):
        try:
            file_path = self.review_interface.current_file_path
            if not file_path:
                InfoBar.warning("Save Error", "No file loaded for saving", parent=self)
                return
            InfoBar.success("Save", f"Changes saved to {os.path.basename(file_path)}", parent=self)
            self._append_log(f"Saved changes to {file_path}")
            
        except Exception as e:
            InfoBar.error("Save Error", str(e), parent=self)
            self._append_log(f"Error saving: {e}")

    def _on_review_retranslate(self, selected_items: list):
        try:
            if not selected_items:
                InfoBar.warning("Retranslate", "No items selected for retranslation", parent=self)
                return
            self._append_log(f"Retranslating {len(selected_items)} selected items")
            InfoBar.success("Retranslate", f"Retranslating {len(selected_items)} items", parent=self)
            
        except Exception as e:
            InfoBar.error("Retranslate Error", str(e), parent=self)
            self._append_log(f"Error retranslating: {e}")

    def closeEvent(self, event):
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.cancel()
                self._worker.wait(5000)
        except Exception:
            pass
        try:
            if self._test_thread is not None and self._test_thread.isRunning():
                self._test_thread.requestInterruption()
                self._test_thread.wait(3000)
        except Exception:
            pass
        event.accept()