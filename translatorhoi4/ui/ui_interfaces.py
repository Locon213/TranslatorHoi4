"""UI interfaces for the main window."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QDesktopServices, QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QScrollArea, QSizePolicy
)

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, PushButton, PrimaryPushButton,
    LineEdit, ComboBox, SpinBox, CheckBox, SwitchButton,
    FluentIcon as FIF, setTheme, Theme,
    InfoBar, InfoBarPosition, ProgressBar, TextEdit,
    CardWidget, SimpleCardWidget, HeaderCardWidget,
    StrongBodyLabel, BodyLabel, SubtitleLabel,
    ScrollArea, MessageBox
)

from .ui_components import SettingCard, SectionHeader, LoadingIndicator
from .ui_threads import IOModelFetchThread
from ..parsers.paradox_yaml import LANG_NAME_LIST, LANG_NATIVE_NAMES, get_native_language_name, parse_source_and_translation
from ..utils.settings import save_settings, load_settings
from ..utils.env import get_api_key, get_cost_currency
from ..utils.logging_config import setup_logging, log_manager
from ..utils.validation import validate_settings, ValidationError
from ..translator.cost import cost_tracker
from ..utils.update_checker import check_for_updates
from ..translator.engine import MODEL_REGISTRY, TranslateWorker, TestModelWorker, JobConfig
from ..utils.fs import collect_localisation_files
from .about import AboutDialog
from .review_window import ReviewInterface
from .translations import translate_text
import os
import json

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
    """Main application window with all interfaces."""

    def __init__(self):
        super().__init__()

        # Setup Window
        self.setWindowTitle("TranslatorHoi4")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.resize(1100, 750)
        self._total_files = 0

        # Theme
        setTheme(Theme.DARK)

        # Initialize Variables & Workers
        self._worker = None
        self._test_thread = None
        self._io_fetch_thread = None
        self._translating = False

        # Create UI Components
        self._init_components()

        # Build Interfaces
        self._init_navigation()

        # Apply logic hooks
        self._switch_backend_settings(self.cmb_model.currentText())

        # Setup UI language change handler
        self.cmb_ui_lang.currentIndexChanged.connect(self._on_ui_lang_changed)

        # Load saved settings
        loaded = self._load_settings()

        # Load API keys from .env as defaults
        self._load_env_defaults()

        # Initialize with default English translation
        current_lang_code = self.cmb_ui_lang.currentData() or 'english'
        self._apply_translations(current_lang_code)

        # Check for updates
        self._check_updates_async()

    def _init_components(self):
        """Initialize all input widgets."""

        # Basic
        self.ed_src = LineEdit()
        self.ed_src.setPlaceholderText("Path to mod folder")
        self.ed_out = LineEdit()
        self.ed_out.setPlaceholderText("Output folder (leave empty for in-place)")
        self.ed_prev = LineEdit()
        self.ed_prev.setPlaceholderText("Optional: previous translation folder")

        self.chk_inplace = CheckBox("Translate in-place (overwrite)")
        self.chk_inplace.stateChanged.connect(self._toggle_inplace)

        self.chk_use_mod_name = CheckBox("Use mod name folder")
        self.chk_use_mod_name.stateChanged.connect(self._toggle_mod_name)

        self.ed_mod_name = LineEdit()
        self.ed_mod_name.setPlaceholderText("Mod name (optional)")
        self.ed_mod_name.setEnabled(False)

        self.cmb_src_lang = ComboBox()
        self.cmb_src_lang.addItems(LANG_NAME_LIST)
        self.cmb_src_lang.setCurrentText("english")

        self.cmb_dst_lang = ComboBox()
        self.cmb_dst_lang.addItems(LANG_NAME_LIST)
        self.cmb_dst_lang.setCurrentText("russian")

        # UI Language selector
        self.cmb_ui_lang = ComboBox()
        for code in LANG_NAME_LIST:
            native_name = get_native_language_name(code)
            self.cmb_ui_lang.addItem(native_name)
            self.cmb_ui_lang.setItemData(self.cmb_ui_lang.count() - 1, code)

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

        # Batch Translation Mode
        self.chk_batch_mode = CheckBox("Batch Translation Mode")
        self.chk_batch_mode.setChecked(False)
        self.spn_chunk_size = SpinBox()
        self.spn_chunk_size.setRange(1, 200)
        self.spn_chunk_size.setValue(100)

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

        # Advanced
        self.chk_strip_md = CheckBox("Strip Markdown")
        self.chk_strip_md.setChecked(True)
        self.chk_rename_files = CheckBox("Auto-rename files (*_l_russian.yml)")
        self.chk_rename_files.setChecked(True)

        self.ed_key_skip = LineEdit()
        self.ed_key_skip.setPlaceholderText("Regex: ^STATE_")

        self.spn_batch = SpinBox(); self.spn_batch.setRange(1, 200); self.spn_batch.setValue(24)
        self.spn_files_cc = SpinBox(); self.spn_files_cc.setRange(1, 6); self.spn_files_cc.setValue(2)
        self.spn_rpm = SpinBox(); self.spn_rpm.setRange(1, 1000); self.spn_rpm.setValue(60)

        # G4F
        self.ed_g4f_model = LineEdit(); self.ed_g4f_model.setText("gpt-4o")
        self.ed_g4f_api_key = LineEdit(); self.ed_g4f_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.chk_g4f_async = CheckBox("Use Async"); self.chk_g4f_async.setChecked(True)
        self.spn_g4f_cc = SpinBox(); self.spn_g4f_cc.setRange(1, 50); self.spn_g4f_cc.setValue(12)

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
        self.spn_io_cc = SpinBox(); self.spn_io_cc.setValue(12)

        # OpenAI
        self.ed_openai_api_key = LineEdit(); self.ed_openai_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_openai_base = LineEdit()
        self.ed_openai_model = LineEdit(); self.ed_openai_model.setPlaceholderText("gpt-4")
        self.chk_openai_async = CheckBox("Use Async"); self.chk_openai_async.setChecked(True)
        self.spn_openai_cc = SpinBox(); self.spn_openai_cc.setValue(12)

        # Anthropic
        self.ed_anthropic_api_key = LineEdit(); self.ed_anthropic_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_anthropic_model = LineEdit(); self.ed_anthropic_model.setPlaceholderText("claude-sonnet-4-5-20250929")
        self.chk_anthropic_async = CheckBox("Use Async"); self.chk_anthropic_async.setChecked(True)
        self.spn_anthropic_cc = SpinBox(); self.spn_anthropic_cc.setValue(12)

        # Gemini
        self.ed_gemini_api_key = LineEdit(); self.ed_gemini_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_gemini_model = LineEdit(); self.ed_gemini_model.setPlaceholderText("gemini-2.5-flash")
        self.chk_gemini_async = CheckBox("Use Async"); self.chk_gemini_async.setChecked(True)
        self.spn_gemini_cc = SpinBox(); self.spn_gemini_cc.setValue(12)

        # Yandex Translate
        self.ed_yandex_translate_api_key = LineEdit(); self.ed_yandex_translate_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_yandex_iam_token = LineEdit(); self.ed_yandex_iam_token.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_yandex_folder_id = LineEdit(); self.ed_yandex_folder_id.setPlaceholderText("b1g20dtckjkooop0futg")
        self.chk_yandex_translate_async = CheckBox("Use Async"); self.chk_yandex_translate_async.setChecked(True)
        self.spn_yandex_translate_cc = SpinBox(); self.spn_yandex_translate_cc.setValue(12)

        # Yandex Cloud
        self.ed_yandex_cloud_api_key = LineEdit(); self.ed_yandex_cloud_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_yandex_cloud_model = LineEdit(); self.ed_yandex_cloud_model.setPlaceholderText("aliceai-llm/latest")
        self.chk_yandex_async = CheckBox("Use Async"); self.chk_yandex_async.setChecked(True)
        self.spn_yandex_cc = SpinBox(); self.spn_yandex_cc.setValue(12)

        # DeepL
        self.ed_deepl_api_key = LineEdit(); self.ed_deepl_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.chk_deepl_async = CheckBox("Use Async"); self.chk_deepl_async.setChecked(True)
        self.spn_deepl_cc = SpinBox(); self.spn_deepl_cc.setValue(12)

        # Fireworks
        self.ed_fireworks_api_key = LineEdit(); self.ed_fireworks_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_fireworks_model = LineEdit(); self.ed_fireworks_model.setPlaceholderText("accounts/fireworks/models/llama-v3p1-8b-instruct")
        self.chk_fireworks_async = CheckBox("Use Async"); self.chk_fireworks_async.setChecked(True)
        self.spn_fireworks_cc = SpinBox(); self.spn_fireworks_cc.setValue(12)

        # Groq
        self.ed_groq_api_key = LineEdit(); self.ed_groq_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_groq_model = LineEdit(); self.ed_groq_model.setPlaceholderText("openai/gpt-oss-20b")
        self.chk_groq_async = CheckBox("Use Async"); self.chk_groq_async.setChecked(True)
        self.spn_groq_cc = SpinBox(); self.spn_groq_cc.setValue(12)

        # Together
        self.ed_together_api_key = LineEdit(); self.ed_together_api_key.setEchoMode(LineEdit.EchoMode.Password)
        self.ed_together_model = LineEdit(); self.ed_together_model.setPlaceholderText("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        self.chk_together_async = CheckBox("Use Async"); self.chk_together_async.setChecked(True)
        self.spn_together_cc = SpinBox(); self.spn_together_cc.setValue(12)

        # Ollama
        self.ed_ollama_model = LineEdit(); self.ed_ollama_model.setPlaceholderText("llama3.2")
        self.ed_ollama_base_url = LineEdit(); self.ed_ollama_base_url.setPlaceholderText("http://localhost:11434")
        self.chk_ollama_async = CheckBox("Use Async"); self.chk_ollama_async.setChecked(True)
        self.spn_ollama_cc = SpinBox(); self.spn_ollama_cc.setValue(12)

        # Tools
        self.ed_glossary = LineEdit()
        self.ed_cache = LineEdit()
        self.cmb_cache_type = ComboBox()
        self.cmb_cache_type.addItems(["SQLite", "JSON"])
        self.cmb_cache_type.setCurrentText("SQLite")

        # Cost Configuration
        self.cmb_currency = ComboBox()
        self.cmb_currency.addItems(["USD", "EUR", "RUB", "GBP"])
        self.cmb_currency.setCurrentText("USD")

        self.g4f_input = LineEdit(); self.g4f_input.setText("0.0")
        self.g4f_output = LineEdit(); self.g4f_output.setText("0.0")
        self.openai_input = LineEdit(); self.openai_input.setText("2.50")
        self.openai_output = LineEdit(); self.openai_output.setText("10.00")
        self.anthropic_input = LineEdit(); self.anthropic_input.setText("3.00")
        self.anthropic_output = LineEdit(); self.anthropic_output.setText("15.00")
        self.gemini_input = LineEdit(); self.gemini_input.setText("0.125")
        self.gemini_output = LineEdit(); self.gemini_output.setText("0.375")
        self.io_input = LineEdit(); self.io_input.setText("0.59")
        self.io_output = LineEdit(); self.io_output.setText("0.79")
        self.yandex_translate_input = LineEdit(); self.yandex_translate_input.setText("0.0")
        self.yandex_translate_output = LineEdit(); self.yandex_translate_output.setText("0.0")
        self.yandex_cloud_input = LineEdit(); self.yandex_cloud_input.setText("0.0")
        self.yandex_cloud_output = LineEdit(); self.yandex_cloud_output.setText("0.0")
        self.deepl_input = LineEdit(); self.deepl_input.setText("0.0")
        self.deepl_output = LineEdit(); self.deepl_output.setText("0.0")
        self.fireworks_input = LineEdit(); self.fireworks_input.setText("0.0")
        self.fireworks_output = LineEdit(); self.fireworks_output.setText("0.0")
        self.groq_input = LineEdit(); self.groq_input.setText("0.0")
        self.groq_output = LineEdit(); self.groq_output.setText("0.0")
        self.together_input = LineEdit(); self.together_input.setText("0.0")
        self.together_output = LineEdit(); self.together_output.setText("0.0")
        self.ollama_input = LineEdit(); self.ollama_input.setText("0.0")
        self.ollama_output = LineEdit(); self.ollama_output.setText("0.0")

        # Monitor (Logs)
        self.pb_global = ProgressBar()
        self.pb_file = ProgressBar()
        self.lbl_stats = BodyLabel("Words: 0 | Keys: 0 | Files: 0/0")
        self.lbl_file = BodyLabel("Ready")
        self.txt_log = TextEdit()
        self.txt_log.setReadOnly(True)

    def _init_navigation(self):
        """Initialize navigation and interfaces."""

        # Home Interface
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
        l_out.addWidget(self.chk_use_mod_name)
        l_out.addWidget(self.ed_mod_name)
        self.home_interface.vBoxLayout.addWidget(card_out)

        # Section: Settings
        self.home_interface.vBoxLayout.addWidget(SectionHeader("General Settings"))

        # UI Language
        self.home_interface.vBoxLayout.addWidget(SettingCard("Interface Language", self.cmb_ui_lang))

        row_lang = QHBoxLayout()
        row_lang.addWidget(SettingCard("Source Language", self.cmb_src_lang))
        row_lang.addWidget(SettingCard("Target Language", self.cmb_dst_lang))
        self.home_interface.vBoxLayout.addLayout(row_lang)

        self.home_interface.vBoxLayout.addWidget(SettingCard("AI Model", self.cmb_model))

        row_params = QHBoxLayout()
        row_params.addWidget(SettingCard("Temperature x100", self.spn_temp))
        row_params.addWidget(SettingCard("Reuse #LOC!", self.chk_reuse_prev))
        self.home_interface.vBoxLayout.addLayout(row_params)

        # Batch Translation Mode
        self.home_interface.vBoxLayout.addWidget(self.chk_batch_mode)
        self.home_interface.vBoxLayout.addWidget(SettingCard("Chunk Size", self.spn_chunk_size))

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

        # Advanced Interface
        self.adv_interface = BaseInterface("AdvancedInterface", self)

        # Processing
        self.adv_interface.vBoxLayout.addWidget(SectionHeader("Processing Rules"))
        proc_card = CardWidget()
        proc_l = QVBoxLayout(proc_card)
        proc_l.addWidget(self.chk_strip_md)
        proc_l.addWidget(self.chk_rename_files)
        h_skip = QHBoxLayout(); h_skip.addWidget(BodyLabel("Skip Regex:")); h_skip.addWidget(self.ed_key_skip)
        proc_l.addLayout(h_skip)
        proc_l.addWidget(SettingCard("RPM Limit", self.spn_rpm))
        proc_l.addWidget(SettingCard("Files Concurrency", self.spn_files_cc))
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
        l.addWidget(self.chk_io_async)
        l.addWidget(SettingCard("Concurrency", self.spn_io_cc))
        self.adv_interface.vBoxLayout.addWidget(self.io_container)

        # OpenAI
        self.openai_container = CardWidget()
        l = QVBoxLayout(self.openai_container)
        l.addWidget(StrongBodyLabel("OpenAI Compatible"))
        l.addWidget(SettingCard("Base URL", self.ed_openai_base))
        l.addWidget(SettingCard("API Key", self.ed_openai_api_key))
        l.addWidget(SettingCard("Model ID", self.ed_openai_model))
        l.addWidget(self.chk_openai_async)
        l.addWidget(SettingCard("Concurrency", self.spn_openai_cc))
        self.adv_interface.vBoxLayout.addWidget(self.openai_container)

        # Anthropic
        self.anthropic_container = CardWidget()
        l = QVBoxLayout(self.anthropic_container)
        l.addWidget(StrongBodyLabel("Anthropic (Claude)"))
        l.addWidget(SettingCard("API Key", self.ed_anthropic_api_key))
        l.addWidget(SettingCard("Model", self.ed_anthropic_model))
        l.addWidget(self.chk_anthropic_async)
        l.addWidget(SettingCard("Concurrency", self.spn_anthropic_cc))
        self.adv_interface.vBoxLayout.addWidget(self.anthropic_container)

        # Gemini
        self.gemini_container = CardWidget()
        l = QVBoxLayout(self.gemini_container)
        l.addWidget(StrongBodyLabel("Google Gemini"))
        l.addWidget(SettingCard("API Key", self.ed_gemini_api_key))
        l.addWidget(SettingCard("Model", self.ed_gemini_model))
        l.addWidget(self.chk_gemini_async)
        l.addWidget(SettingCard("Concurrency", self.spn_gemini_cc))
        self.adv_interface.vBoxLayout.addWidget(self.gemini_container)

        # Yandex Translate
        self.yandex_translate_container = CardWidget()
        l = QVBoxLayout(self.yandex_translate_container)
        l.addWidget(StrongBodyLabel("Yandex Translate"))
        l.addWidget(SettingCard("API Key", self.ed_yandex_translate_api_key))
        l.addWidget(SettingCard("IAM Token", self.ed_yandex_iam_token))
        l.addWidget(SettingCard("Folder ID", self.ed_yandex_folder_id))
        btn_yandex_iam = PushButton("Get IAM Token", self, FIF.LINK)
        btn_yandex_iam.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://yandex.cloud/ru/docs/iam/operations/iam-token/create")))
        btn_yandex_api = PushButton("Get API Key", self, FIF.LINK)
        btn_yandex_api.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://yandex.cloud/ru/docs/iam/concepts/authorization/api-key")))
        h_yandex_btns = QHBoxLayout()
        h_yandex_btns.addWidget(btn_yandex_iam)
        h_yandex_btns.addWidget(btn_yandex_api)
        l.addLayout(h_yandex_btns)
        l.addWidget(self.chk_yandex_translate_async)
        l.addWidget(SettingCard("Concurrency", self.spn_yandex_translate_cc))
        self.adv_interface.vBoxLayout.addWidget(self.yandex_translate_container)

        # Yandex Cloud
        self.yandex_cloud_container = CardWidget()
        l = QVBoxLayout(self.yandex_cloud_container)
        l.addWidget(StrongBodyLabel("Yandex Cloud"))
        l.addWidget(SettingCard("API Key", self.ed_yandex_cloud_api_key))
        l.addWidget(SettingCard("Model", self.ed_yandex_cloud_model))
        l.addWidget(SettingCard("Folder ID", self.ed_yandex_folder_id))
        l.addWidget(self.chk_yandex_async)
        l.addWidget(SettingCard("Concurrency", self.spn_yandex_cc))
        self.adv_interface.vBoxLayout.addWidget(self.yandex_cloud_container)

        # DeepL
        self.deepl_container = CardWidget()
        l = QVBoxLayout(self.deepl_container)
        l.addWidget(StrongBodyLabel("DeepL API"))
        l.addWidget(SettingCard("API Key", self.ed_deepl_api_key))
        l.addWidget(self.chk_deepl_async)
        l.addWidget(SettingCard("Concurrency", self.spn_deepl_cc))
        self.adv_interface.vBoxLayout.addWidget(self.deepl_container)

        # Fireworks
        self.fireworks_container = CardWidget()
        l = QVBoxLayout(self.fireworks_container)
        l.addWidget(StrongBodyLabel("Fireworks.ai"))
        l.addWidget(SettingCard("API Key", self.ed_fireworks_api_key))
        l.addWidget(SettingCard("Model", self.ed_fireworks_model))
        l.addWidget(self.chk_fireworks_async)
        l.addWidget(SettingCard("Concurrency", self.spn_fireworks_cc))
        self.adv_interface.vBoxLayout.addWidget(self.fireworks_container)

        # Groq
        self.groq_container = CardWidget()
        l = QVBoxLayout(self.groq_container)
        l.addWidget(StrongBodyLabel("Groq"))
        l.addWidget(SettingCard("API Key", self.ed_groq_api_key))
        l.addWidget(SettingCard("Model", self.ed_groq_model))
        l.addWidget(self.chk_groq_async)
        l.addWidget(SettingCard("Concurrency", self.spn_groq_cc))
        self.adv_interface.vBoxLayout.addWidget(self.groq_container)

        # Together
        self.together_container = CardWidget()
        l = QVBoxLayout(self.together_container)
        l.addWidget(StrongBodyLabel("Together.ai"))
        l.addWidget(SettingCard("API Key", self.ed_together_api_key))
        l.addWidget(SettingCard("Model", self.ed_together_model))
        l.addWidget(self.chk_together_async)
        l.addWidget(SettingCard("Concurrency", self.spn_together_cc))
        self.adv_interface.vBoxLayout.addWidget(self.together_container)

        # Ollama
        self.ollama_container = CardWidget()
        l = QVBoxLayout(self.ollama_container)
        l.addWidget(StrongBodyLabel("Ollama"))
        l.addWidget(SettingCard("Model", self.ed_ollama_model))
        l.addWidget(SettingCard("Base URL", self.ed_ollama_base_url))
        l.addWidget(self.chk_ollama_async)
        l.addWidget(SettingCard("Concurrency", self.spn_ollama_cc))
        self.adv_interface.vBoxLayout.addWidget(self.ollama_container)

        # Tools Interface
        self.tools_interface = BaseInterface("ToolsInterface", self)
        self.tools_interface.vBoxLayout.addWidget(SectionHeader("Data Management"))

        btn_gloss = PushButton("Load CSV"); btn_gloss.clicked.connect(self._pick_glossary)
        self.tools_interface.vBoxLayout.addWidget(SettingCard("Glossary Path", self.ed_glossary))

        btn_clear = PushButton("Clear Cache"); btn_clear.clicked.connect(self._clear_cache)
        self.tools_interface.vBoxLayout.addWidget(SettingCard("Cache File", self.ed_cache))
        self.tools_interface.vBoxLayout.addWidget(SettingCard("Cache Type", self.cmb_cache_type))
        self.tools_interface.vBoxLayout.addWidget(btn_clear)

        btn_check_updates = PushButton("Check for Updates", self, FIF.UPDATE)
        btn_check_updates.clicked.connect(self._check_updates_async)
        self.tools_interface.vBoxLayout.addWidget(btn_check_updates)

        self.tools_interface.vBoxLayout.addWidget(SectionHeader("Cost Configuration"))
        self.tools_interface.vBoxLayout.addWidget(SettingCard("Currency", self.cmb_currency))

        self.tools_interface.vBoxLayout.addWidget(SectionHeader("Cost Rates ($ per million tokens)"))

        # G4F
        g4f_card = CardWidget()
        g4f_l = QVBoxLayout(g4f_card)
        g4f_l.addWidget(StrongBodyLabel("G4F"))
        g4f_l.addWidget(SettingCard("Input Cost", self.g4f_input))
        g4f_l.addWidget(SettingCard("Output Cost", self.g4f_output))
        self.tools_interface.vBoxLayout.addWidget(g4f_card)

        # OpenAI
        openai_card = CardWidget()
        openai_l = QVBoxLayout(openai_card)
        openai_l.addWidget(StrongBodyLabel("OpenAI"))
        openai_l.addWidget(SettingCard("Input Cost", self.openai_input))
        openai_l.addWidget(SettingCard("Output Cost", self.openai_output))
        self.tools_interface.vBoxLayout.addWidget(openai_card)

        # Anthropic
        anthropic_card = CardWidget()
        anthropic_l = QVBoxLayout(anthropic_card)
        anthropic_l.addWidget(StrongBodyLabel("Anthropic"))
        anthropic_l.addWidget(SettingCard("Input Cost", self.anthropic_input))
        anthropic_l.addWidget(SettingCard("Output Cost", self.anthropic_output))
        self.tools_interface.vBoxLayout.addWidget(anthropic_card)

        # Gemini
        gemini_card = CardWidget()
        gemini_l = QVBoxLayout(gemini_card)
        gemini_l.addWidget(StrongBodyLabel("Gemini"))
        gemini_l.addWidget(SettingCard("Input Cost", self.gemini_input))
        gemini_l.addWidget(SettingCard("Output Cost", self.gemini_output))
        self.tools_interface.vBoxLayout.addWidget(gemini_card)

        # IO
        io_card = CardWidget()
        io_l = QVBoxLayout(io_card)
        io_l.addWidget(StrongBodyLabel("IO Intelligence"))
        io_l.addWidget(SettingCard("Input Cost", self.io_input))
        io_l.addWidget(SettingCard("Output Cost", self.io_output))
        self.tools_interface.vBoxLayout.addWidget(io_card)

        # Yandex Translate
        yandex_translate_card = CardWidget()
        yandex_translate_l = QVBoxLayout(yandex_translate_card)
        yandex_translate_l.addWidget(StrongBodyLabel("Yandex Translate"))
        yandex_translate_l.addWidget(SettingCard("Input Cost", self.yandex_translate_input))
        yandex_translate_l.addWidget(SettingCard("Output Cost", self.yandex_translate_output))
        self.tools_interface.vBoxLayout.addWidget(yandex_translate_card)

        # Yandex Cloud
        yandex_cloud_card = CardWidget()
        yandex_cloud_l = QVBoxLayout(yandex_cloud_card)
        yandex_cloud_l.addWidget(StrongBodyLabel("Yandex Cloud"))
        yandex_cloud_l.addWidget(SettingCard("Input Cost", self.yandex_cloud_input))
        yandex_cloud_l.addWidget(SettingCard("Output Cost", self.yandex_cloud_output))
        self.tools_interface.vBoxLayout.addWidget(yandex_cloud_card)

        # DeepL
        deepl_card = CardWidget()
        deepl_l = QVBoxLayout(deepl_card)
        deepl_l.addWidget(StrongBodyLabel("DeepL API"))
        deepl_l.addWidget(SettingCard("Input Cost", self.deepl_input))
        deepl_l.addWidget(SettingCard("Output Cost", self.deepl_output))
        self.tools_interface.vBoxLayout.addWidget(deepl_card)

        # Fireworks
        fireworks_card = CardWidget()
        fireworks_l = QVBoxLayout(fireworks_card)
        fireworks_l.addWidget(StrongBodyLabel("Fireworks.ai"))
        fireworks_l.addWidget(SettingCard("Input Cost", self.fireworks_input))
        fireworks_l.addWidget(SettingCard("Output Cost", self.fireworks_output))
        self.tools_interface.vBoxLayout.addWidget(fireworks_card)

        # Groq
        groq_card = CardWidget()
        groq_l = QVBoxLayout(groq_card)
        groq_l.addWidget(StrongBodyLabel("Groq"))
        groq_l.addWidget(SettingCard("Input Cost", self.groq_input))
        groq_l.addWidget(SettingCard("Output Cost", self.groq_output))
        self.tools_interface.vBoxLayout.addWidget(groq_card)

        # Together
        together_card = CardWidget()
        together_l = QVBoxLayout(together_card)
        together_l.addWidget(StrongBodyLabel("Together.ai"))
        together_l.addWidget(SettingCard("Input Cost", self.together_input))
        together_l.addWidget(SettingCard("Output Cost", self.together_output))
        self.tools_interface.vBoxLayout.addWidget(together_card)

        # Ollama
        ollama_card = CardWidget()
        ollama_l = QVBoxLayout(ollama_card)
        ollama_l.addWidget(StrongBodyLabel("Ollama"))
        ollama_l.addWidget(SettingCard("Input Cost", self.ollama_input))
        ollama_l.addWidget(SettingCard("Output Cost", self.ollama_output))
        self.tools_interface.vBoxLayout.addWidget(ollama_card)

        self.tools_interface.vBoxLayout.addWidget(SectionHeader("Presets"))
        h_preset = QHBoxLayout()
        btn_save = PushButton("Save Preset", self, FIF.SAVE)
        btn_save.clicked.connect(self._save_preset)
        btn_load = PushButton("Load Preset", self, FIF.FOLDER)
        btn_load.clicked.connect(self._load_preset)
        h_preset.addWidget(btn_save); h_preset.addWidget(btn_load)
        self.tools_interface.vBoxLayout.addLayout(h_preset)

        # Monitor Interface
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

        # Add to window
        self.addSubInterface(self.home_interface, FIF.HOME, "Home")
        self.addSubInterface(self.adv_interface, FIF.SETTING, "Advanced Settings")
        self.addSubInterface(self.tools_interface, FIF.DEVELOPER_TOOLS, "Tools")
        self.addSubInterface(self.monitor_interface, FIF.COMMAND_PROMPT, "Process Monitor")

        # Review Interface
        self.review_interface = ReviewInterface(self)
        self.review_interface.src_dir = self.ed_src.text().strip()
        self.review_interface.src_lang = self.cmb_src_lang.currentText()
        self.review_interface.dst_lang = self.cmb_dst_lang.currentText()
        self.addSubInterface(self.review_interface, FIF.EDIT, "Review & Edit")

        # About button
        self.navigationInterface.addItem(
            routeKey="About",
            icon=FIF.INFO,
            text="About",
            onClick=self._show_about,
            selectable=False,
            position=NavigationItemPosition.BOTTOM
        )

    # --- UI TRANSLATION LOGIC ---

    def _on_ui_lang_changed(self):
        """Handle UI language change instantly."""
        # Prevent recursive calls during translation
        if getattr(self, '_translating', False):
            return

        lang_code = self.cmb_ui_lang.currentData()
        if lang_code:
            self._translating = True
            try:
                self._apply_translations(lang_code)
                # Save settings AFTER translations are applied to preserve choice
                self._save_settings()
            except Exception as e:
                print(f"Error applying translations: {e}")
            finally:
                self._translating = False

    def _apply_translations(self, lang_code: str):
        """Apply translations to all UI elements properly handling original text."""

        # 1. Translate Window Title
        self.setWindowTitle(translate_text("TranslatorHoi4", lang_code))

        # 2. Translate Navigation Items (Sidebar)
        self._retranslate_navigation(lang_code)

        # 3. Translate Widgets in all Interfaces
        self._retranslate_widgets(self.home_interface, lang_code)
        self._retranslate_widgets(self.adv_interface, lang_code)
        self._retranslate_widgets(self.tools_interface, lang_code)
        self._retranslate_widgets(self.monitor_interface, lang_code)

        # 4. Review Interface
        if hasattr(self.review_interface, '_apply_translations'):
             self.review_interface._apply_translations(lang_code)
        else:
             self._retranslate_widgets(self.review_interface, lang_code)

        # 5. Force update
        QApplication.processEvents()

    def _retranslate_navigation(self, lang_code: str):
        """Retranslate sidebar items safely without using .widget() lookup."""
        # Mapping of Route Key -> Original English Text
        # Route keys must match what you used in self.addSubInterface or addItem
        route_text_map = {
            "Home": "Home",
            "Advanced Settings": "Advanced Settings",
            "Tools": "Tools",
            "Process Monitor": "Process Monitor",
            "Review & Edit": "Review & Edit",
            "About": "About"
        }

        # Method 1: Try to access internal items dict (QFluentWidgets specific)
        # self.navigationInterface.panel.items usually holds {routeKey: NavigationItem}
        updated_via_panel = False
        if hasattr(self, 'navigationInterface'):
            panel = getattr(self.navigationInterface, 'panel', None)
            if panel and hasattr(panel, 'items'):
                try:
                    for route_key, nav_item in panel.items.items():
                        if route_key in route_text_map:
                            original_text = route_text_map[route_key]
                            translated = translate_text(original_text, lang_code)
                            if hasattr(nav_item, 'setText'):
                                nav_item.setText(translated)
                    updated_via_panel = True
                except Exception as e:
                    print(f"Warning: Failed to update nav via panel items: {e}")

        # Method 2: Fallback - iterate all children widgets and match text
        # This is useful if internal structure changes but text is visible
        if not updated_via_panel:
            for widget in self.navigationInterface.findChildren(QWidget):
                # Skip if it doesn't have text method
                if not hasattr(widget, 'text') or not hasattr(widget, 'setText'):
                    continue

                # Try to match current text or cached original text
                current_text = widget.text()

                # Check if we have cached original text
                original_text = widget.property("original_text")
                if not original_text:
                    # Check if current text is one of our known English keys
                    # This implies we are in English or first run
                    if current_text in route_text_map.values():
                        original_text = current_text
                        widget.setProperty("original_text", original_text)
                    else:
                        # Try to reverse lookup? Hard. Just skip if unknown.
                        continue

                if original_text:
                    translated = translate_text(original_text, lang_code)
                    if translated != current_text:
                        widget.setText(translated)

    def _retranslate_widgets(self, root_widget: QWidget, lang_code: str):
        """Recursively retranslate all labels and buttons in a widget."""
        # findChildren with QWidget finds ALL descendants recursively
        for child in root_widget.findChildren(QWidget):
            self._translate_single_widget(child, lang_code)

    def _translate_single_widget(self, widget: QWidget, lang_code: str):
        """
        Translates a single widget using cached original text.
        """
        if not widget: return
        if hasattr(widget, 'isValid') and not widget.isValid(): return

        properties_to_translate = [
            ("text", "setText"),
            ("placeholderText", "setPlaceholderText")
        ]

        for prop_name, setter_name in properties_to_translate:
            if not hasattr(widget, prop_name) or not hasattr(widget, setter_name):
                continue

            try:
                getter = getattr(widget, prop_name)
                current_val = getter()
            except (RuntimeError, AttributeError):
                continue

            if not isinstance(current_val, str) or not current_val:
                continue

            cache_key = f"original_{prop_name}"
            original_text = widget.property(cache_key)

            if original_text is None:
                # Logic for ComboBoxes inside widgets
                if hasattr(widget, 'text') and widget.text() == "Interface Language":
                    original_text = "Interface Language"
                elif isinstance(widget, ComboBox):
                    # Cache items for ComboBox
                    for i in range(widget.count()):
                        item_data = widget.itemData(i)
                        # Specific check for language combo boxes
                        if item_data and isinstance(item_data, str) and item_data in LANG_NAME_LIST:
                            original_item_text = get_native_language_name(item_data)
                            translated_item_text = translate_text(original_item_text, lang_code)
                            widget.setItemText(i, translated_item_text)
                    # ComboBox main text usually follows current item, no need to set property usually
                    # unless editable.
                    continue
                else:
                    original_text = current_val
                widget.setProperty(cache_key, original_text)

            translated_text = translate_text(original_text, lang_code)

            if translated_text != current_val:
                setter = getattr(widget, setter_name)
                setter(translated_text)

    def _on_ui_lang_changed(self):
        """Handle UI language change instantly."""
        # Prevent recursive calls during translation
        if getattr(self, '_translating', False):
            return

        lang_code = self.cmb_ui_lang.currentData()
        if lang_code:
            self._translating = True
            try:
                self._apply_translations(lang_code)
                # Save settings AFTER translations are applied to preserve choice
                self._save_settings()
            except Exception as e:
                print(f"Error applying translations: {e}")
            finally:
                self._translating = False

    def _apply_translations(self, lang_code: str):
        """Apply translations to all UI elements properly handling original text."""

        # 1. Translate Window Title
        self.setWindowTitle(translate_text("TranslatorHoi4", lang_code))

        # 2. Translate Navigation Items (Sidebar)
        self._retranslate_navigation(lang_code)

        # 3. Translate Widgets in all Interfaces
        self._retranslate_widgets(self.home_interface, lang_code)
        self._retranslate_widgets(self.adv_interface, lang_code)
        self._retranslate_widgets(self.tools_interface, lang_code)
        self._retranslate_widgets(self.monitor_interface, lang_code)

        # 4. Force update
        QApplication.processEvents()

    def _retranslate_navigation(self, lang_code: str):
        """Retranslate sidebar items safely without using .widget() lookup."""
        # Mapping of Route Key -> Original English Text
        # Route keys must match what you used in self.addSubInterface or addItem
        route_text_map = {
            "Home": "Home",
            "Advanced Settings": "Advanced Settings",
            "Tools": "Tools",
            "Process Monitor": "Process Monitor",
            "Review & Edit": "Review & Edit",
            "About": "About"
        }

        # Method 1: Try to access internal items dict (QFluentWidgets specific)
        # self.navigationInterface.panel.items usually holds {routeKey: NavigationItem}
        updated_via_panel = False
        if hasattr(self, 'navigationInterface'):
            panel = getattr(self.navigationInterface, 'panel', None)
            if panel and hasattr(panel, 'items'):
                try:
                    for route_key, nav_item in panel.items.items():
                        if route_key in route_text_map:
                            original_text = route_text_map[route_key]
                            translated = translate_text(original_text, lang_code)
                            if hasattr(nav_item, 'setText'):
                                nav_item.setText(translated)
                    updated_via_panel = True
                except Exception as e:
                    print(f"Warning: Failed to update nav via panel items: {e}")

        # Method 2: Fallback - iterate all children widgets and match text
        # This is useful if internal structure changes but text is visible
        if not updated_via_panel:
            for widget in self.navigationInterface.findChildren(QWidget):
                # Skip if it doesn't have text method
                if not hasattr(widget, 'text') or not hasattr(widget, 'setText'):
                    continue

                # Try to match current text or cached original text
                current_text = widget.text()

                # Check if we have cached original text
                original_text = widget.property("original_text")
                if not original_text:
                    # Check if current text is one of our known English keys
                    # This implies we are in English or first run
                    if current_text in route_text_map.values():
                        original_text = current_text
                        widget.setProperty("original_text", original_text)
                    else:
                        # Try to reverse lookup? Hard. Just skip if unknown.
                        continue

                if original_text:
                    translated = translate_text(original_text, lang_code)
                    if translated != current_text:
                        widget.setText(translated)

    def _retranslate_widgets(self, root_widget: QWidget, lang_code: str):
        """Recursively retranslate all labels and buttons in a widget."""
        # findChildren with QWidget finds ALL descendants recursively
        for child in root_widget.findChildren(QWidget):
            self._translate_single_widget(child, lang_code)

    def _translate_single_widget(self, widget: QWidget, lang_code: str):
        """
        Translates a single widget using cached original text.
        """
        if not widget: return
        if hasattr(widget, 'isValid') and not widget.isValid(): return

        properties_to_translate = [
            ("text", "setText"),
            ("placeholderText", "setPlaceholderText")
        ]

        for prop_name, setter_name in properties_to_translate:
            if not hasattr(widget, prop_name) or not hasattr(widget, setter_name):
                continue

            try:
                getter = getattr(widget, prop_name)
                current_val = getter()
            except (RuntimeError, AttributeError):
                continue

            if not isinstance(current_val, str) or not current_val:
                continue

            cache_key = f"original_{prop_name}"
            original_text = widget.property(cache_key)

            if original_text is None:
                # Logic for ComboBoxes inside widgets
                if hasattr(widget, 'text') and widget.text() == "Interface Language":
                    original_text = "Interface Language"
                elif isinstance(widget, ComboBox):
                    # Cache items for ComboBox
                    for i in range(widget.count()):
                        item_data = widget.itemData(i)
                        # Specific check for language combo boxes
                        if item_data and isinstance(item_data, str) and item_data in LANG_NAME_LIST:
                            original_item_text = get_native_language_name(item_data)
                            translated_item_text = translate_text(original_item_text, lang_code)
                            widget.setItemText(i, translated_item_text)
                    # ComboBox main text usually follows current item, no need to set property usually
                    # unless editable.
                    continue
                else:
                    original_text = current_val
                widget.setProperty(cache_key, original_text)

            translated_text = translate_text(original_text, lang_code)

            if translated_text != current_val:
                setter = getattr(widget, setter_name)
                setter(translated_text)

    # --- END UI TRANSLATION LOGIC ---

    def _save_settings(self):
        """Save current settings to cache file."""
        ui_lang = self.cmb_ui_lang.currentData()
        if not ui_lang:
            ui_lang = 'english'

        data = {
            'src': self.ed_src.text(),
            'out': self.ed_out.text(),
            'prev': self.ed_prev.text(),
            'in_place': self.chk_inplace.isChecked(),
            'src_lang': self.cmb_src_lang.currentText(),
            'dst_lang': self.cmb_dst_lang.currentText(),
            'ui_lang': ui_lang,
            'model': self.cmb_model.currentText(),
            'temp_x100': self.spn_temp.value(),
            'skip_existing': self.chk_skip_exist.isChecked(),
            'strip_md': self.chk_strip_md.isChecked(),
            'rename_files': self.chk_rename_files.isChecked(),
            'key_skip_regex': self.ed_key_skip.text(),
            'batch_size': self.spn_batch.value(),
            'files_cc': self.spn_files_cc.value(),
            'glossary': self.ed_glossary.text(),
            'cache': self.ed_cache.text(),
            'cache_type': self.cmb_cache_type.currentText(),
            'reuse_prev_loc': self.chk_reuse_prev.isChecked(),
            'mark_loc': self.chk_mark_loc.isChecked(),
            'batch_translation': self.chk_batch_mode.isChecked(),
            'chunk_size': self.spn_chunk_size.value(),
            'g4f_model': self.ed_g4f_model.text(),
            'g4f_api_key': self.ed_g4f_api_key.text(),
            'g4f_async': self.chk_g4f_async.isChecked(),
            'g4f_cc': self.spn_g4f_cc.value(),
            'io_api_key': self.ed_io_api_key.text(),
            'io_base_url': self.ed_io_base.text(),
            'io_async': self.chk_io_async.isChecked(),
            'io_cc': self.spn_io_cc.value(),
            'openai_api_key': self.ed_openai_api_key.text(),
            'openai_base_url': self.ed_openai_base.text(),
            'openai_model': self.ed_openai_model.text(),
            'openai_async': self.chk_openai_async.isChecked(),
            'openai_cc': self.spn_openai_cc.value(),
            'anthropic_api_key': self.ed_anthropic_api_key.text(),
            'anthropic_model': self.ed_anthropic_model.text(),
            'anthropic_async': self.chk_anthropic_async.isChecked(),
            'anthropic_cc': self.spn_anthropic_cc.value(),
            'gemini_api_key': self.ed_gemini_api_key.text(),
            'gemini_model': self.ed_gemini_model.text(),
            'gemini_async': self.chk_gemini_async.isChecked(),
            'gemini_cc': self.spn_gemini_cc.value(),
            'yandex_translate_api_key': self.ed_yandex_translate_api_key.text(),
            'yandex_iam_token': self.ed_yandex_iam_token.text(),
            'yandex_folder_id': self.ed_yandex_folder_id.text(),
            'yandex_cloud_api_key': self.ed_yandex_cloud_api_key.text(),
            'yandex_cloud_model': self.ed_yandex_cloud_model.text(),
            'yandex_async': self.chk_yandex_async.isChecked(),
            'yandex_cc': self.spn_yandex_cc.value(),
            'deepl_api_key': self.ed_deepl_api_key.text(),
            'fireworks_api_key': self.ed_fireworks_api_key.text(),
            'fireworks_model': self.ed_fireworks_model.text(),
            'fireworks_async': self.chk_fireworks_async.isChecked(),
            'fireworks_cc': self.spn_fireworks_cc.value(),
            'groq_api_key': self.ed_groq_api_key.text(),
            'groq_model': self.ed_groq_model.text(),
            'groq_async': self.chk_groq_async.isChecked(),
            'groq_cc': self.spn_groq_cc.value(),
            'together_api_key': self.ed_together_api_key.text(),
            'together_model': self.ed_together_model.text(),
            'together_async': self.chk_together_async.isChecked(),
            'together_cc': self.spn_together_cc.value(),
            'ollama_model': self.ed_ollama_model.text(),
            'ollama_base_url': self.ed_ollama_base_url.text(),
            'ollama_async': self.chk_ollama_async.isChecked(),
            'ollama_cc': self.spn_ollama_cc.value(),
            'currency': self.cmb_currency.currentText(),
            'g4f_input_cost': self.g4f_input.text(),
            'g4f_output_cost': self.g4f_output.text(),
            'openai_input_cost': self.openai_input.text(),
            'openai_output_cost': self.openai_output.text(),
            'anthropic_input_cost': self.anthropic_input.text(),
            'anthropic_output_cost': self.anthropic_output.text(),
            'gemini_input_cost': self.gemini_input.text(),
            'gemini_output_cost': self.gemini_output.text(),
            'io_input_cost': self.io_input.text(),
            'io_output_cost': self.io_output.text(),
            'yandex_translate_input_cost': self.yandex_translate_input.text(),
            'yandex_translate_output_cost': self.yandex_translate_output.text(),
            'yandex_cloud_input_cost': self.yandex_cloud_input.text(),
            'yandex_cloud_output_cost': self.yandex_cloud_output.text(),
            'deepl_input_cost': self.deepl_input.text(),
            'deepl_output_cost': self.deepl_output.text(),
            'fireworks_input_cost': self.fireworks_input.text(),
            'fireworks_output_cost': self.fireworks_output.text(),
            'groq_input_cost': self.groq_input.text(),
            'groq_output_cost': self.groq_output.text(),
            'together_input_cost': self.together_input.text(),
            'together_output_cost': self.together_output.text(),
            'ollama_input_cost': self.ollama_input.text(),
            'ollama_output_cost': self.ollama_output.text(),
        }
        save_settings(data)

    def _load_settings(self):
        """Load settings from cache file."""
        settings = load_settings()
        if not settings:
            return False

        try:
            # Path settings
            if settings.get('src'): self.ed_src.setText(settings['src'])
            if settings.get('out'): self.ed_out.setText(settings['out'])
            if settings.get('prev'): self.ed_prev.setText(settings['prev'])

            # Checkboxes
            self.chk_inplace.setChecked(bool(settings.get('in_place', False)))
            self.chk_skip_exist.setChecked(bool(settings.get('skip_existing', False)))
            self.chk_reuse_prev.setChecked(bool(settings.get('reuse_prev_loc', True)))
            self.chk_mark_loc.setChecked(bool(settings.get('mark_loc', True)))
            self.chk_batch_mode.setChecked(bool(settings.get('batch_translation', False)))

            # Language settings
            src_lang = settings.get('src_lang', 'english')
            dst_lang = settings.get('dst_lang', 'russian')
            ui_lang = settings.get('ui_lang', 'english')

            if src_lang in LANG_NAME_LIST:
                self.cmb_src_lang.setCurrentText(src_lang)
            if dst_lang in LANG_NAME_LIST:
                self.cmb_dst_lang.setCurrentText(dst_lang)

            # Set UI language without triggering signal immediately
            self.cmb_ui_lang.blockSignals(True)
            idx = self.cmb_ui_lang.findData(ui_lang)
            if idx >= 0:
                self.cmb_ui_lang.setCurrentIndex(idx)
            else:
                idx = self.cmb_ui_lang.findData('english')
                if idx >= 0: self.cmb_ui_lang.setCurrentIndex(idx)
            self.cmb_ui_lang.blockSignals(False)

            # Model settings
            model = settings.get('model')
            if model and model in MODEL_REGISTRY:
                self.cmb_model.setCurrentText(model)

            # Temperature
            temp = settings.get('temp_x100', 70)
            if isinstance(temp, (int, float)):
                self.spn_temp.setValue(int(temp))

            # Chunk size
            chunk_size = settings.get('chunk_size', 50)
            if isinstance(chunk_size, (int, float)):
                self.spn_chunk_size.setValue(int(chunk_size))

            # Advanced settings
            self.chk_strip_md.setChecked(bool(settings.get('strip_md', True)))
            self.chk_rename_files.setChecked(bool(settings.get('rename_files', True)))
            if settings.get('key_skip_regex'):
                self.ed_key_skip.setText(settings['key_skip_regex'])

            self.spn_batch.setValue(int(settings.get('batch_size', 12)))
            self.spn_files_cc.setValue(int(settings.get('files_cc', 1)))

            # G4F settings
            if settings.get('g4f_model'): self.ed_g4f_model.setText(settings['g4f_model'])
            if settings.get('g4f_api_key'): self.ed_g4f_api_key.setText(settings['g4f_api_key'])
            self.chk_g4f_async.setChecked(bool(settings.get('g4f_async', True)))
            self.spn_g4f_cc.setValue(int(settings.get('g4f_cc', 6)))

            # IO settings
            if settings.get('io_api_key'): self.ed_io_api_key.setText(settings['io_api_key'])
            if settings.get('io_base_url'): self.ed_io_base.setText(settings['io_base_url'])
            self.chk_io_async.setChecked(bool(settings.get('io_async', True)))
            self.spn_io_cc.setValue(int(settings.get('io_cc', 6)))

            # OpenAI settings
            if settings.get('openai_api_key'): self.ed_openai_api_key.setText(settings['openai_api_key'])
            if settings.get('openai_base_url'): self.ed_openai_base.setText(settings['openai_base_url'])
            if settings.get('openai_model'): self.ed_openai_model.setText(settings['openai_model'])
            self.chk_openai_async.setChecked(bool(settings.get('openai_async', True)))
            self.spn_openai_cc.setValue(int(settings.get('openai_cc', 6)))

            # Anthropic settings
            if settings.get('anthropic_api_key'): self.ed_anthropic_api_key.setText(settings['anthropic_api_key'])
            if settings.get('anthropic_model'): self.ed_anthropic_model.setText(settings['anthropic_model'])
            self.chk_anthropic_async.setChecked(bool(settings.get('anthropic_async', True)))
            self.spn_anthropic_cc.setValue(int(settings.get('anthropic_cc', 6)))

            # Gemini settings
            if settings.get('gemini_api_key'): self.ed_gemini_api_key.setText(settings['gemini_api_key'])
            if settings.get('gemini_model'): self.ed_gemini_model.setText(settings['gemini_model'])
            self.chk_gemini_async.setChecked(bool(settings.get('gemini_async', True)))
            self.spn_gemini_cc.setValue(int(settings.get('gemini_cc', 6)))

            # Yandex Translate settings
            if settings.get('yandex_translate_api_key'): self.ed_yandex_translate_api_key.setText(settings['yandex_translate_api_key'])
            if settings.get('yandex_iam_token'): self.ed_yandex_iam_token.setText(settings['yandex_iam_token'])
            if settings.get('yandex_folder_id'): self.ed_yandex_folder_id.setText(settings['yandex_folder_id'])

            # Yandex Cloud settings
            if settings.get('yandex_cloud_api_key'): self.ed_yandex_cloud_api_key.setText(settings['yandex_cloud_api_key'])
            if settings.get('yandex_cloud_model'): self.ed_yandex_cloud_model.setText(settings['yandex_cloud_model'])
            self.chk_yandex_async.setChecked(bool(settings.get('yandex_async', True)))
            self.spn_yandex_cc.setValue(int(settings.get('yandex_cc', 6)))

            # DeepL settings
            if settings.get('deepl_api_key'): self.ed_deepl_api_key.setText(settings['deepl_api_key'])

            # Fireworks settings
            if settings.get('fireworks_api_key'): self.ed_fireworks_api_key.setText(settings['fireworks_api_key'])
            if settings.get('fireworks_model'): self.ed_fireworks_model.setText(settings['fireworks_model'])
            self.chk_fireworks_async.setChecked(bool(settings.get('fireworks_async', True)))
            self.spn_fireworks_cc.setValue(int(settings.get('fireworks_cc', 6)))

            # Groq settings
            if settings.get('groq_api_key'): self.ed_groq_api_key.setText(settings['groq_api_key'])
            if settings.get('groq_model'): self.ed_groq_model.setText(settings['groq_model'])
            self.chk_groq_async.setChecked(bool(settings.get('groq_async', True)))
            self.spn_groq_cc.setValue(int(settings.get('groq_cc', 6)))

            # Together settings
            if settings.get('together_api_key'): self.ed_together_api_key.setText(settings['together_api_key'])
            if settings.get('together_model'): self.ed_together_model.setText(settings['together_model'])
            self.chk_together_async.setChecked(bool(settings.get('together_async', True)))
            self.spn_together_cc.setValue(int(settings.get('together_cc', 6)))

            # Ollama settings
            if settings.get('ollama_model'): self.ed_ollama_model.setText(settings['ollama_model'])
            if settings.get('ollama_base_url'): self.ed_ollama_base_url.setText(settings['ollama_base_url'])
            self.chk_ollama_async.setChecked(bool(settings.get('ollama_async', True)))
            self.spn_ollama_cc.setValue(int(settings.get('ollama_cc', 6)))

            # Tools settings
            if settings.get('glossary'): self.ed_glossary.setText(settings['glossary'])
            if settings.get('cache'): self.ed_cache.setText(settings['cache'])
            self.cmb_cache_type.setCurrentText(settings.get('cache_type', 'SQLite'))

            # Cost settings
            currency = settings.get('currency', 'USD')
            if currency in ["USD", "EUR", "RUB", "GBP"]:
                self.cmb_currency.setCurrentText(currency)

            self.g4f_input.setText(settings.get('g4f_input_cost', '0.0'))
            self.g4f_output.setText(settings.get('g4f_output_cost', '0.0'))
            self.openai_input.setText(settings.get('openai_input_cost', '2.50'))
            self.openai_output.setText(settings.get('openai_output_cost', '10.00'))
            self.anthropic_input.setText(settings.get('anthropic_input_cost', '3.00'))
            self.anthropic_output.setText(settings.get('anthropic_output_cost', '15.00'))
            self.gemini_input.setText(settings.get('gemini_input_cost', '0.125'))
            self.gemini_output.setText(settings.get('gemini_output_cost', '0.375'))
            self.io_input.setText(settings.get('io_input_cost', '0.59'))
            self.io_output.setText(settings.get('io_output_cost', '0.79'))
            self.yandex_translate_input.setText(settings.get('yandex_translate_input_cost', '0.0'))
            self.yandex_translate_output.setText(settings.get('yandex_translate_output_cost', '0.0'))
            self.yandex_cloud_input.setText(settings.get('yandex_cloud_input_cost', '0.0'))
            self.yandex_cloud_output.setText(settings.get('yandex_cloud_output_cost', '0.0'))
            self.deepl_input.setText(settings.get('deepl_input_cost', '0.0'))
            self.deepl_output.setText(settings.get('deepl_output_cost', '0.0'))
            self.fireworks_input.setText(settings.get('fireworks_input_cost', '0.0'))
            self.fireworks_output.setText(settings.get('fireworks_output_cost', '0.0'))
            self.groq_input.setText(settings.get('groq_input_cost', '0.0'))
            self.groq_output.setText(settings.get('groq_output_cost', '0.0'))
            self.together_input.setText(settings.get('together_input_cost', '0.0'))
            self.together_output.setText(settings.get('together_output_cost', '0.0'))
            self.ollama_input.setText(settings.get('ollama_input_cost', '0.0'))
            self.ollama_output.setText(settings.get('ollama_output_cost', '0.0'))

            # Update inplace UI state
            self._toggle_inplace()

        except Exception as e:
            log_manager.error(f"Failed to load settings: {e}")
            return False

        return True

    def _load_env_defaults(self):
        """Load default values from .env file if not set in settings."""
        # Load API keys from .env if fields are empty
        if not self.ed_g4f_api_key.text().strip():
            api_key = get_api_key('g4f')
            if api_key:
                self.ed_g4f_api_key.setText(api_key)

        if not self.ed_io_api_key.text().strip():
            api_key = get_api_key('io')
            if api_key:
                self.ed_io_api_key.setText(api_key)

        if not self.ed_openai_api_key.text().strip():
            api_key = get_api_key('openai')
            if api_key:
                self.ed_openai_api_key.setText(api_key)

        if not self.ed_anthropic_api_key.text().strip():
            api_key = get_api_key('anthropic')
            if api_key:
                self.ed_anthropic_api_key.setText(api_key)

        if not self.ed_gemini_api_key.text().strip():
            api_key = get_api_key('gemini')
            if api_key:
                self.ed_gemini_api_key.setText(api_key)

        # Set cost currency
        currency = get_cost_currency()
        if currency:
            cost_tracker.set_currency(currency)

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
        self.yandex_translate_container.setVisible(text == "Yandex Translate")
        self.yandex_cloud_container.setVisible(text == "Yandex Cloud")
        self.deepl_container.setVisible(text == "DeepL API")
        self.fireworks_container.setVisible(text == "Fireworks.ai")
        self.groq_container.setVisible(text == "Groq")
        self.together_container.setVisible(text == "Together.ai")
        self.ollama_container.setVisible(text == "Ollama")

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

    def _toggle_mod_name(self):
        v = self.chk_use_mod_name.isChecked()
        self.ed_mod_name.setEnabled(v)

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
            p = os.path.join(out, ".hoi4loc_cache") if out else ""
        # Adjust path based on cache type
        cache_type = self.cmb_cache_type.currentText().lower()
        if cache_type == "sqlite" and p and not p.endswith('.db'):
            p += '.db'
        elif cache_type == "json" and p and not p.endswith('.json'):
            p += '.json'
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
            "ui_lang": self.cmb_ui_lang.currentData(),  # FIX: Save UI language
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
            "batch_translation": self.chk_batch_mode.isChecked(),
            "chunk_size": self.spn_chunk_size.value(),
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

            # FIX: Load UI language properly
            ui_lang = data.get("ui_lang", "english")
            self.cmb_ui_lang.blockSignals(True)
            idx = self.cmb_ui_lang.findData(ui_lang)
            if idx >= 0:
                self.cmb_ui_lang.setCurrentIndex(idx)
            self.cmb_ui_lang.blockSignals(False)

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
            self.chk_batch_mode.setChecked(bool(data.get("batch_translation", False)))
            self.spn_chunk_size.setValue(int(data.get("chunk_size", 50)))
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

            # Apply translations if UI lang changed
            if data.get("ui_lang"):
                 self._apply_translations(data.get("ui_lang"))

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
            yandex_translate_api_key=self.ed_yandex_translate_api_key.text().strip() or None,
            yandex_iam_token=self.ed_yandex_iam_token.text().strip() or None,
            yandex_folder_id=self.ed_yandex_folder_id.text().strip(),
            yandex_cloud_api_key=self.ed_yandex_cloud_api_key.text().strip() or None,
            yandex_cloud_model=self.ed_yandex_cloud_model.text().strip() or "aliceai-llm/latest",
            yandex_async=self.chk_yandex_async.isChecked(),
            yandex_concurrency=self.spn_yandex_cc.value(),
            deepl_api_key=self.ed_deepl_api_key.text().strip() or None,
            fireworks_api_key=self.ed_fireworks_api_key.text().strip() or None,
            fireworks_model=self.ed_fireworks_model.text().strip() or "accounts/fireworks/models/llama-v3p1-8b-instruct",
            fireworks_async=self.chk_fireworks_async.isChecked(),
            fireworks_concurrency=self.spn_fireworks_cc.value(),
            groq_api_key=self.ed_groq_api_key.text().strip() or None,
            groq_model=self.ed_groq_model.text().strip() or "openai/gpt-oss-20b",
            groq_async=self.chk_groq_async.isChecked(),
            groq_concurrency=self.spn_groq_cc.value(),
            together_api_key=self.ed_together_api_key.text().strip() or None,
            together_model=self.ed_together_model.text().strip() or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            together_async=self.chk_together_async.isChecked(),
            together_concurrency=self.spn_together_cc.value(),
            ollama_model=self.ed_ollama_model.text().strip() or "llama3.2",
            ollama_base_url=self.ed_ollama_base_url.text().strip() or "http://localhost:11434",
            ollama_async=self.chk_ollama_async.isChecked(),
            ollama_concurrency=self.spn_ollama_cc.value(),
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

        # Validate settings before starting
        try:
            settings = {
                'src': self.ed_src.text().strip(),
                'out': self.ed_out.text().strip() if not self.chk_inplace.isChecked() else self.ed_src.text().strip(),
                'src_lang': self.cmb_src_lang.currentText(),
                'dst_lang': self.cmb_dst_lang.currentText(),
                'model': self.cmb_model.currentText(),
                'temp_x100': self.spn_temp.value(),
                'batch_size': self.spn_batch.value(),
                'files_cc': self.spn_files_cc.value(),
                'g4f_cc': self.spn_g4f_cc.value(),
                'io_cc': self.spn_io_cc.value(),
                'openai_cc': self.spn_openai_cc.value(),
                'anthropic_cc': self.spn_anthropic_cc.value(),
                'gemini_cc': self.spn_gemini_cc.value(),
                'key_skip_regex': self.ed_key_skip.text().strip(),
            }
            validated_settings = validate_settings(settings)
        except ValidationError as e:
            InfoBar.error("Validation Error", str(e), parent=self)
            return

        src = validated_settings['src']
        out = validated_settings['out']
        in_place = self.chk_inplace.isChecked()

        cache_path = (self.ed_cache.text().strip() or None)
        if cache_path is None:
            base = out or src
            cache_path = os.path.join(base, ".hoi4loc_cache")
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
            batch_translation=self.chk_batch_mode.isChecked(),
            chunk_size=self.spn_chunk_size.value(),
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
        from ..utils.fs import collect_localisation_files
        try:
            out_dir = self.ed_out.text().strip()
            if not out_dir:
                out_dir = self.ed_src.text().strip()

            if not out_dir:
                return

            src_dir = self.ed_src.text().strip()
            if not src_dir:
                src_dir = out_dir

            files = collect_localisation_files(out_dir)
            if not files:
                return

            file_path = files[0]
            self._append_log(f"Loading {file_path} for review")

            # Find corresponding source file
            src_lang = self.cmb_src_lang.currentText()
            dst_lang = self.cmb_dst_lang.currentText()
            rel_path = os.path.relpath(file_path, out_dir)
            # Convert filename to source language format
            expected_src_basename = os.path.basename(rel_path).replace(f"_l_{dst_lang}", f"_l_{src_lang}")
            src_file_path = None
            for src_file in collect_localisation_files(src_dir):
                if os.path.basename(src_file) == expected_src_basename:
                    src_file_path = src_file
                    break

            # Use combined parsing to get proper original/translation
            data = parse_source_and_translation(src_file_path, file_path)

            if data:
                self.review_interface.load_data(file_path, data, files)
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

    def _check_updates_async(self):
        """Check for updates asynchronously."""
        from PyQt6.QtCore import QThread, pyqtSignal

        class UpdateCheckThread(QThread):
            finished = pyqtSignal(dict)

            def run(self):
                try:
                    update_info = check_for_updates()
                    self.finished.emit(update_info)
                except Exception as e:
                    log_manager.error(f"Update check failed: {e}")
                    self.finished.emit({"update_available": False})

        self._update_thread = UpdateCheckThread()
        self._update_thread.finished.connect(self._on_update_check_finished)
        self._update_thread.start()

    def _on_update_check_finished(self, update_info):
        """Handle update check results."""
        if update_info.get("update_available"):
            version = update_info.get("latest_version", "unknown")
            InfoBar.info(
                title="Update Available",
                content=f"Version {version} is available. Check the About dialog for download link.",
                parent=self
            )
            log_manager.info(f"Update available: {version}")

    def closeEvent(self, event):
        # Save settings before closing
        self._save_settings()

        # End cost tracking session
        session_summary = cost_tracker.end_session()
        if session_summary.get("total_cost", 0) > 0:
            log_manager.info(f"Session cost: ${session_summary['total_cost']:.4f} {session_summary.get('currency', 'USD')}")

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
        try:
            if hasattr(self, '_update_thread') and self._update_thread.isRunning():
                self._update_thread.wait(2000)
        except Exception:
            pass
        event.accept()