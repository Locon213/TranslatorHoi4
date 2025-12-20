"""Main application window."""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from PyQt6.QtCore import Qt, QSize, QUrl, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QDesktopServices, QIcon, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QTextEdit, QComboBox, QSpinBox,
    QGroupBox, QFormLayout, QCheckBox, QTabWidget
)

from ..parsers.paradox_yaml import LANG_NAME_LIST
from ..utils.fs import collect_localisation_files
from ..utils.version import __version__
from ..translator.engine import JobConfig, MODEL_REGISTRY, TranslateWorker, TestModelWorker
from .theme import DARK_QSS
from .about import AboutDialog


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

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        pen = QPen(QColor(85, 85, 85))
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HOI4 Localizer ✨")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.setMinimumSize(QSize(1100, 800))
        self._worker: Optional[TranslateWorker] = None
        self._test_thread: Optional[TestModelWorker] = None
        self._io_fetch_thread: Optional[IOModelFetchThread] = None
        self._total_files = 0
        self._build_ui()
        self._apply_dark_theme()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        # --- Basic tab ---
        tab_basic = QWidget()
        basic_layout = QVBoxLayout(tab_basic); basic_layout.setContentsMargins(6,6,6,6); basic_layout.setSpacing(8)
        path_box = QGroupBox("Paths"); path_form = QFormLayout()
        self.ed_src = QLineEdit(); btn_src = QPushButton("Browse…"); btn_src.clicked.connect(self._pick_src)
        h1 = QHBoxLayout(); h1.addWidget(self.ed_src); h1.addWidget(btn_src); w1 = QWidget(); w1.setLayout(h1)
        path_form.addRow(QLabel("Source mod folder:"), w1)
        self.ed_out = QLineEdit(); btn_out = QPushButton("Browse…"); btn_out.clicked.connect(self._pick_out)
        h2 = QHBoxLayout(); h2.addWidget(self.ed_out); h2.addWidget(btn_out); w2 = QWidget(); w2.setLayout(h2)
        path_form.addRow(QLabel("Output folder (blank = in-place):"), w2)
        open_out_btn = QPushButton("Open output folder"); open_out_btn.clicked.connect(self._open_out)
        path_form.addRow(open_out_btn)
        self.chk_inplace = QCheckBox("Translate in-place (overwrite files)"); self.chk_inplace.stateChanged.connect(self._toggle_inplace)
        path_form.addRow(self.chk_inplace)
        self.ed_prev = QLineEdit(); self.ed_prev.setPlaceholderText("Optional: previous localized folder to reuse #LOC!")
        btn_prev = QPushButton("Browse…"); btn_prev.clicked.connect(self._pick_prev)
        hp = QHBoxLayout(); hp.addWidget(self.ed_prev); hp.addWidget(btn_prev); wp = QWidget(); wp.setLayout(hp)
        path_form.addRow(QLabel("Previous localized folder:"), wp)
        path_box.setLayout(path_form)
        set_box = QGroupBox("Basic settings"); set_form = QFormLayout()
        self.cmb_src_lang = QComboBox(); self.cmb_src_lang.addItems(LANG_NAME_LIST); self.cmb_src_lang.setCurrentText("english")
        self.cmb_dst_lang = QComboBox(); self.cmb_dst_lang.addItems(LANG_NAME_LIST); self.cmb_dst_lang.setCurrentText("russian")
        set_form.addRow("From:", self.cmb_src_lang)
        set_form.addRow("To:", self.cmb_dst_lang)
        self.cmb_model = QComboBox(); self.cmb_model.addItems(list(MODEL_REGISTRY.keys()))
        self.cmb_model.setCurrentText("G4F: chat.completions")
        self.cmb_model.currentTextChanged.connect(self._switch_backend_settings)
        set_form.addRow("Model:", self.cmb_model)
        temp_row = QHBoxLayout()
        self.spn_temp = QSpinBox(); self.spn_temp.setRange(1, 120); self.spn_temp.setValue(70)
        temp_row.addWidget(QLabel("Temperature ×100:")); temp_row.addWidget(self.spn_temp); temp_row.addStretch(1)
        temp_w = QWidget(); temp_w.setLayout(temp_row)
        set_form.addRow("Model sampling:", temp_w)
        self.chk_skip_exist = QCheckBox("Skip files that already exist in output"); self.chk_skip_exist.setChecked(False)
        set_form.addRow(self.chk_skip_exist)
        self.chk_mark_loc = QCheckBox("Mark translated lines with #LOC!"); self.chk_mark_loc.setChecked(True)
        set_form.addRow(self.chk_mark_loc)
        self.chk_reuse_prev = QCheckBox("Reuse previous #LOC! translations"); self.chk_reuse_prev.setChecked(True)
        set_form.addRow(self.chk_reuse_prev)
        set_box.setLayout(set_form)
        basic_layout.addWidget(path_box)
        basic_layout.addWidget(set_box)
        tabs.addTab(tab_basic, "Basic")

        # Advanced tab
        tab_adv = QWidget()
        adv_layout = QVBoxLayout(tab_adv); adv_layout.setContentsMargins(6,6,6,6); adv_layout.setSpacing(8)
        adv_box = QGroupBox("Advanced"); adv_form = QFormLayout()
        self.ed_hf_token = QLineEdit(); self.ed_hf_token.setPlaceholderText("Optional — HF token (hf_...)"); self.ed_hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        adv_form.addRow("HF Token:", self.ed_hf_token)
        self.ed_hf_direct = QLineEdit(); self.ed_hf_direct.setPlaceholderText("Optional: direct HF URL (https://owner-space.hf.space)")
        adv_form.addRow("Direct HF URL:", self.ed_hf_direct)
        self.chk_strip_md = QCheckBox("Strip model Markdown/analysis"); self.chk_strip_md.setChecked(True); adv_form.addRow(self.chk_strip_md)
        self.chk_rename_files = QCheckBox("Rename filenames to target language (e.g., *_l_russian.yml)"); self.chk_rename_files.setChecked(True); adv_form.addRow(self.chk_rename_files)
        self.ed_key_skip = QLineEdit(); self.ed_key_skip.setPlaceholderText(r"Optional key skip regex, e.g. ^STATE_")
        adv_form.addRow("Skip keys (regex):", self.ed_key_skip)
        adv_box.setLayout(adv_form)
        perf_box = QGroupBox("Performance"); perf_form = QFormLayout()
        self.spn_batch = QSpinBox(); self.spn_batch.setRange(1, 200); self.spn_batch.setValue(12)
        perf_form.addRow("Batch size (Google/G4F):", self.spn_batch)
        self.spn_hf_cc = QSpinBox(); self.spn_hf_cc.setRange(1, 8); self.spn_hf_cc.setValue(2)
        perf_form.addRow("HF concurrency (per file):", self.spn_hf_cc)
        self.spn_files_cc = QSpinBox(); self.spn_files_cc.setRange(1, 6); self.spn_files_cc.setValue(1)
        perf_form.addRow("Files concurrency:", self.spn_files_cc)
        perf_box.setLayout(perf_form)
        self.g4f_box = QGroupBox("G4F settings"); g4f_form = QFormLayout()
        self.ed_g4f_model = QLineEdit(); self.ed_g4f_model.setPlaceholderText("gemini-2.5-flash"); self.ed_g4f_model.setText("gemini-2.5-flash")
        g4f_form.addRow("Model:", self.ed_g4f_model)
        self.ed_g4f_provider = QLineEdit(); self.ed_g4f_provider.setPlaceholderText("e.g. g4f.Provider.Blackbox")
        g4f_form.addRow("Provider:", self.ed_g4f_provider)
        self.ed_g4f_api_key = QLineEdit(); self.ed_g4f_api_key.setPlaceholderText("Optional API key"); self.ed_g4f_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        g4f_form.addRow("API key:", self.ed_g4f_api_key)
        self.ed_g4f_proxies = QLineEdit(); self.ed_g4f_proxies.setPlaceholderText("Optional proxies, http://user:pass@host")
        g4f_form.addRow("Proxies:", self.ed_g4f_proxies)
        self.chk_g4f_async = QCheckBox("Use AsyncClient"); self.chk_g4f_async.setChecked(True)
        g4f_form.addRow(self.chk_g4f_async)
        self.spn_g4f_cc = QSpinBox(); self.spn_g4f_cc.setRange(1, 50); self.spn_g4f_cc.setValue(6)
        g4f_form.addRow("G4F concurrency:", self.spn_g4f_cc)
        self.chk_g4f_web = QCheckBox("Enable web_search"); self.chk_g4f_web.setChecked(False)
        g4f_form.addRow(self.chk_g4f_web)
        self.g4f_box.setLayout(g4f_form)

        self.io_box = QGroupBox("IO Intelligence settings"); io_form = QFormLayout()
        self.ed_io_api_key = QLineEdit(); self.ed_io_api_key.setPlaceholderText("API key"); self.ed_io_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        io_form.addRow("API key:", self.ed_io_api_key)
        self.ed_io_base = QLineEdit(); self.ed_io_base.setPlaceholderText("https://api.intelligence.io.solutions/api/v1/")
        io_form.addRow("Base URL:", self.ed_io_base)
        self.cmb_io_model = QComboBox(); self.cmb_io_model.addItem("Please wait…"); self.cmb_io_model.setEnabled(False)
        self.io_loader = LoadingIndicator(); self.io_loader.hide()
        h_io_model = QHBoxLayout(); h_io_model.addWidget(self.cmb_io_model); h_io_model.addWidget(self.io_loader); w_io_model = QWidget(); w_io_model.setLayout(h_io_model)
        io_form.addRow("Model:", w_io_model)
        self.chk_io_async = QCheckBox("Use AsyncClient"); self.chk_io_async.setChecked(True); io_form.addRow(self.chk_io_async)
        self.spn_io_cc = QSpinBox(); self.spn_io_cc.setRange(1, 50); self.spn_io_cc.setValue(6); io_form.addRow("IO concurrency:", self.spn_io_cc)
        self.io_box.setLayout(io_form)

        self.openai_box = QGroupBox("OpenAI Compatible settings"); openai_form = QFormLayout()
        self.ed_openai_api_key = QLineEdit(); self.ed_openai_api_key.setPlaceholderText("API key"); self.ed_openai_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        openai_form.addRow("API key:", self.ed_openai_api_key)
        self.ed_openai_base = QLineEdit(); self.ed_openai_base.setPlaceholderText("https://api.openai.com/v1/")
        openai_form.addRow("Base URL:", self.ed_openai_base)
        self.ed_openai_model = QLineEdit(); self.ed_openai_model.setPlaceholderText("gpt-4")
        openai_form.addRow("Model:", self.ed_openai_model)
        self.chk_openai_async = QCheckBox("Use AsyncClient"); self.chk_openai_async.setChecked(True)
        openai_form.addRow(self.chk_openai_async)
        self.spn_openai_cc = QSpinBox(); self.spn_openai_cc.setRange(1, 50); self.spn_openai_cc.setValue(6)
        openai_form.addRow("OpenAI concurrency:", self.spn_openai_cc)
        self.openai_box.setLayout(openai_form)

        adv_layout.addWidget(adv_box)
        adv_layout.addWidget(perf_box)
        adv_layout.addWidget(self.g4f_box)
        adv_layout.addWidget(self.io_box)
        adv_layout.addWidget(self.openai_box)
        self.io_box.hide()
        self.openai_box.hide()
        tabs.addTab(tab_adv, "Advanced")

        # Tools tab
        tab_tools = QWidget()
        tools_layout = QVBoxLayout(tab_tools); tools_layout.setContentsMargins(6,6,6,6); tools_layout.setSpacing(8)
        gloss_box = QGroupBox("Glossary"); gloss_form = QFormLayout()
        self.ed_glossary = QLineEdit(); btn_gl = QPushButton("Load CSV…"); btn_gl.clicked.connect(self._pick_glossary)
        hg = QHBoxLayout(); hg.addWidget(self.ed_glossary); hg.addWidget(btn_gl); wg = QWidget(); wg.setLayout(hg)
        gloss_form.addRow(QLabel("CSV (mode,src,dst):"), wg)
        gloss_box.setLayout(gloss_form)
        cache_box = QGroupBox("Cache"); cache_form = QFormLayout()
        self.ed_cache = QLineEdit(); self.ed_cache.setPlaceholderText(".hoi4loc_cache.json (auto if empty)")
        cache_form.addRow("Cache file path:", self.ed_cache)
        btn_clear_cache = QPushButton("Clear cache file"); btn_clear_cache.clicked.connect(self._clear_cache)
        cache_form.addRow(btn_clear_cache)
        cache_box.setLayout(cache_form)
        preset_box = QGroupBox("Presets"); preset_form = QFormLayout()
        btn_save = QPushButton("Save preset…"); btn_save.clicked.connect(self._save_preset)
        btn_load = QPushButton("Load preset…"); btn_load.clicked.connect(self._load_preset)
        hh = QHBoxLayout(); hh.addWidget(btn_save); hh.addWidget(btn_load); wpr = QWidget(); wpr.setLayout(hh)
        preset_form.addRow(wpr)
        preset_box.setLayout(preset_form)
        tools_layout.addWidget(gloss_box)
        tools_layout.addWidget(cache_box)
        tools_layout.addWidget(preset_box)
        tabs.addTab(tab_tools, "Tools")

        root.addWidget(tabs)

        act_row = QHBoxLayout()
        self.btn_scan = QPushButton("Scan files"); self.btn_scan.clicked.connect(self._scan_files)
        self.btn_test = QPushButton("Test model"); self.btn_test.clicked.connect(self._test_model_async)
        self.btn_go = QPushButton("Start Translating ✨"); self.btn_go.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.clicked.connect(self._cancel); self.btn_cancel.setEnabled(False)
        act_row.addWidget(self.btn_scan); act_row.addStretch(1); act_row.addWidget(self.btn_test); act_row.addWidget(self.btn_go); act_row.addWidget(self.btn_cancel)
        root.addLayout(act_row)

        g_row = QHBoxLayout()
        self.pb_global = QProgressBar(); self.pb_global.setMinimum(0); self.pb_global.setMaximum(100)
        g_row.addWidget(QLabel("All files:")); g_row.addWidget(self.pb_global, 1)
        self.lbl_stats = QLabel("Words: 0 | Keys: 0 | Files: 0/0"); g_row.addWidget(self.lbl_stats)
        root.addLayout(g_row)

        f_row = QHBoxLayout()
        self.lbl_file = QLabel("")
        f_row.addWidget(QLabel("Current file:")); f_row.addWidget(self.lbl_file, 3)
        self.pb_file = QProgressBar(); self.pb_file.setMinimum(0); self.pb_file.setMaximum(100)
        f_row.addWidget(self.pb_file, 2)
        root.addLayout(f_row)

        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        root.addWidget(self.txt_log, 1)
        self._make_menu()
        self._switch_backend_settings(self.cmb_model.currentText())

    def _make_menu(self):
        bar = self.menuBar()
        filem = bar.addMenu("File")
        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.close)
        filem.addAction(act_quit)
        them = bar.addMenu("Theme")
        act_dark = QAction("Dark", self); act_dark.triggered.connect(self._apply_dark_theme)
        act_light = QAction("Light", self); act_light.triggered.connect(self._apply_light_theme)
        them.addAction(act_dark); them.addAction(act_light)
        helpm = bar.addMenu("Help")
        about_act = QAction("About", self); about_act.triggered.connect(self._show_about)
        helpm.addAction(about_act)

    # slots/utils
    def _show_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def _switch_backend_settings(self, text: str):
        self.g4f_box.setVisible(text == "G4F: chat.completions")
        self.io_box.setVisible(text == "IO: chat.completions")
        self.openai_box.setVisible(text == "OpenAI Compatible API")
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
            QMessageBox.warning(self, "Scan", "Please choose a valid source folder"); return
        files = collect_localisation_files(src)
        self._total_files = len(files)
        self._append_log(f"Found {len(files)} localisation files.")
        self.lbl_stats.setText(f"Words: 0 | Keys: 0 | Files: 0/{self._total_files}")

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
            except Exception as e:
                self._append_log(f"Failed to clear cache: {e}")
        else:
            self._append_log("No cache file found.")

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
            "hf_token": self.ed_hf_token.text(),
            "hf_direct": self.ed_hf_direct.text(),
            "strip_md": self.chk_strip_md.isChecked(),
            "rename_files": self.chk_rename_files.isChecked(),
            "key_skip_regex": self.ed_key_skip.text(),
            "batch_size": self.spn_batch.value(),
            "hf_cc": self.spn_hf_cc.value(),
            "files_cc": self.spn_files_cc.value(),
            "glossary": self.ed_glossary.text(),
            "cache": self.ed_cache.text(),
            "reuse_prev_loc": self.chk_reuse_prev.isChecked(),
            "mark_loc": self.chk_mark_loc.isChecked(),
            "g4f_model": self.ed_g4f_model.text(),
            "g4f_provider": self.ed_g4f_provider.text(),
            "g4f_api_key": self.ed_g4f_api_key.text(),
            "g4f_proxies": self.ed_g4f_proxies.text(),
            "g4f_async": self.chk_g4f_async.isChecked(),
            "g4f_cc": self.spn_g4f_cc.value(),
            "g4f_web_search": self.chk_g4f_web.isChecked(),
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
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._append_log(f"Preset saved → {p}")
        except Exception as e:
            QMessageBox.critical(self, "Preset", f"Failed to save: {e}")

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
            model = data.get("model", "G4F: chat.completions")
            if model in MODEL_REGISTRY: self.cmb_model.setCurrentText(model)
            self.spn_temp.setValue(int(data.get("temp_x100", 70)))
            self.chk_skip_exist.setChecked(bool(data.get("skip_existing", False)))
            self.ed_hf_token.setText(data.get("hf_token",""))
            self.ed_hf_direct.setText(data.get("hf_direct",""))
            self.chk_strip_md.setChecked(bool(data.get("strip_md", True)))
            self.chk_rename_files.setChecked(bool(data.get("rename_files", True)))
            self.ed_key_skip.setText(data.get("key_skip_regex",""))
            self.spn_batch.setValue(int(data.get("batch_size", 12)))
            self.spn_hf_cc.setValue(int(data.get("hf_cc", 2)))
            self.spn_files_cc.setValue(int(data.get("files_cc", 1)))
            self.ed_glossary.setText(data.get("glossary",""))
            self.ed_cache.setText(data.get("cache",""))
            self.chk_reuse_prev.setChecked(bool(data.get("reuse_prev_loc", True)))
            self.chk_mark_loc.setChecked(bool(data.get("mark_loc", True)))
            self.ed_g4f_model.setText(data.get("g4f_model","gemini-2.5-flash"))
            self.ed_g4f_provider.setText(data.get("g4f_provider",""))
            self.ed_g4f_api_key.setText(data.get("g4f_api_key",""))
            self.ed_g4f_proxies.setText(data.get("g4f_proxies",""))
            self.chk_g4f_async.setChecked(bool(data.get("g4f_async", True)))
            self.spn_g4f_cc.setValue(int(data.get("g4f_cc", 6)))
            self.chk_g4f_web.setChecked(bool(data.get("g4f_web_search", False)))
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
            self._append_log(f"Preset loaded ← {p}")
        except Exception as e:
            QMessageBox.critical(self, "Preset", f"Failed to load: {e}")

    def _test_model_async(self):
        if self._test_thread is not None:
            QMessageBox.information(self, "Test", "Test is already running."); return
        self.btn_test.setEnabled(False)
        self._test_thread = TestModelWorker(
            model_key=self.cmb_model.currentText(),
            src_lang=self.cmb_src_lang.currentText(),
            dst_lang=self.cmb_dst_lang.currentText(),
            temperature=self.spn_temp.value() / 100.0,
            hf_token=self.ed_hf_token.text().strip() or None,
            hf_direct_url=self.ed_hf_direct.text().strip() or None,
            strip_md=self.chk_strip_md.isChecked(),
            glossary_path=self.ed_glossary.text().strip() or None,
            g4f_model=self.ed_g4f_model.text().strip() or None,
            g4f_provider=self.ed_g4f_provider.text().strip() or None,
            g4f_api_key=self.ed_g4f_api_key.text().strip() or None,
            g4f_proxies=self.ed_g4f_proxies.text().strip() or None,
            g4f_async=self.chk_g4f_async.isChecked(),
            g4f_concurrency=self.spn_g4f_cc.value(),
            g4f_web_search=self.chk_g4f_web.isChecked(),
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
        )
        self._test_thread.ok.connect(self._on_test_ok)
        self._test_thread.fail.connect(self._on_test_fail)
        self._test_thread.finished.connect(self._on_test_finished, Qt.ConnectionType.QueuedConnection)
        self._test_thread.start()

    def _on_test_ok(self, txt: str):
        self._append_log(f"Test OK → {txt}")

    def _on_test_fail(self, e: str):
        self._append_log(f"Test failed: {e}")

    def _on_test_finished(self):
        self.btn_test.setEnabled(True)
        self._test_thread = None

    def _start(self):
        src = self.ed_src.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Start", "Please choose a valid source folder"); return
        in_place = self.chk_inplace.isChecked()
        out = self.ed_out.text().strip()
        if not in_place and (not out or not os.path.isdir(out)):
            QMessageBox.warning(self, "Start", "Please choose a valid output folder or enable in-place mode"); return
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
            hf_token=(self.ed_hf_token.text().strip() or None),
            hf_direct_url=(self.ed_hf_direct.text().strip() or None),
            in_place=in_place,
            skip_existing=self.chk_skip_exist.isChecked(),
            strip_md=self.chk_strip_md.isChecked(),
            batch_size=self.spn_batch.value(),
            rename_files=self.chk_rename_files.isChecked(),
            hf_concurrency=self.spn_hf_cc.value(),
            files_concurrency=self.spn_files_cc.value(),
            key_skip_regex=(self.ed_key_skip.text().strip() or None),
            cache_path=cache_path,
            glossary_path=(self.ed_glossary.text().strip() or None),
            prev_loc_dir=(self.ed_prev.text().strip() or None),
            reuse_prev_loc=self.chk_reuse_prev.isChecked(),
            mark_loc_flag=self.chk_mark_loc.isChecked(),
            g4f_model=self.ed_g4f_model.text().strip() or "gemini-2.5-flash",
            g4f_provider=self.ed_g4f_provider.text().strip() or None,
            g4f_api_key=self.ed_g4f_api_key.text().strip() or None,
            g4f_proxies=self.ed_g4f_proxies.text().strip() or None,
            g4f_async=self.chk_g4f_async.isChecked(),
            g4f_concurrency=self.spn_g4f_cc.value(),
            g4f_web_search=self.chk_g4f_web.isChecked(),
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
        )
        if cfg.model_key == "G4F: chat.completions":
            os.environ["G4F_MODEL"] = (cfg.g4f_model or "gemini-2.5-flash")
            os.environ["G4F_PROVIDER"] = (cfg.g4f_provider or "")
            os.environ["G4F_API_KEY"] = (cfg.g4f_api_key or "")
            os.environ["G4F_PROXIES"] = (cfg.g4f_proxies or "")
            os.environ["G4F_TEMP"] = str(cfg.temperature)
            os.environ["G4F_ASYNC"] = "1" if cfg.g4f_async else "0"
            os.environ["G4F_CONCURRENCY"] = str(cfg.g4f_concurrency)
            os.environ["G4F_WEB_SEARCH"] = "1" if cfg.g4f_web_search else "0"
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
        self._append_log(
            f"Starting with {cfg.model_key} (temp={cfg.temperature}, files_cc={cfg.files_concurrency})…"
        )
        self.btn_go.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.pb_global.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_file.setText("")
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
        self.btn_go.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(2000)
        except Exception:
            pass
        self._worker = None

    def _on_aborted(self, msg: str):
        self._append_log(f"Aborted: {msg}")
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

    def _apply_dark_theme(self):
        self.setStyleSheet(DARK_QSS)

    def _apply_light_theme(self):
        self.setStyleSheet("")
