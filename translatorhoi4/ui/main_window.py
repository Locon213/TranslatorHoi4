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
from ..utils.env import get_api_key, get_cost_currency
from ..utils.logging_config import setup_logging, log_manager
from ..utils.validation import validate_settings, ValidationError
from ..translator.cost import cost_tracker
from ..translator.engine import MODEL_REGISTRY, TestModelWorker, JobConfig, TranslateWorker, RetranslateWorker
from ..utils.update_checker import check_for_updates
from ..utils.settings import save_settings, load_settings
from ..utils.version import __version__
from ..utils.fs import collect_localisation_files
from .ui_components import SettingCard, SectionHeader, LoadingIndicator
from .ui_threads import IOModelFetchThread
from .ui_interfaces import BaseInterface, MainWindow as BaseMainWindow
from ..parsers.paradox_yaml import LANG_NAME_LIST, LANG_NATIVE_NAMES, get_native_language_name, parse_source_and_translation
from .about import AboutDialog
from .review_window import ReviewInterface
from .translations import translate_text

class MainWindow(BaseMainWindow):
    def __init__(self):
        super().__init__()

        # 1. Setup Window
        self.setWindowTitle("TranslatorHoi4")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.resize(1100, 750)
        self._total_files = 0

        # 2. Theme
        setTheme(Theme.DARK)

        # 3. Initialize Variables & Workers
        self._worker: Optional[TranslateWorker] = None
        self._test_thread: Optional[TestModelWorker] = None
        self._io_fetch_thread: Optional[IOModelFetchThread] = None
        self._retranslate_worker: Optional[RetranslateWorker] = None
        self._translating = False  # Flag to prevent recursive translation calls

        # 4. Setup logging
        setup_logging()

        # 5. Initialize cost tracker
        cost_tracker.start_session()

        # 6. Apply logic hooks
        self._switch_backend_settings(self.cmb_model.currentText())

        # 7. Setup UI language change handler
        # Connect this LAST to avoid triggering during initialization
        self.cmb_ui_lang.currentIndexChanged.connect(self._on_ui_lang_changed)

        # 8. Load saved settings AFTER interfaces are built
        loaded = self._load_settings()

        # 9. Load API keys from .env as defaults if not set in settings
        self._load_env_defaults()

        # 10. Set cost configuration
        cost_tracker.set_currency(self.cmb_currency.currentText())
        cost_tracker.set_cost_per_million('g4f', float(self.g4f_input.text()), float(self.g4f_output.text()))
        cost_tracker.set_cost_per_million('openai', float(self.openai_input.text()), float(self.openai_output.text()))
        cost_tracker.set_cost_per_million('anthropic', float(self.anthropic_input.text()), float(self.anthropic_output.text()))
        cost_tracker.set_cost_per_million('gemini', float(self.gemini_input.text()), float(self.gemini_output.text()))
        cost_tracker.set_cost_per_million('io', float(self.io_input.text()), float(self.io_output.text()))
        cost_tracker.set_cost_per_million('yandex_translate', float(self.yandex_translate_input.text()), float(self.yandex_translate_output.text()))
        cost_tracker.set_cost_per_million('yandex_cloud', float(self.yandex_cloud_input.text()), float(self.yandex_cloud_output.text()))
        cost_tracker.set_cost_per_million('deepl', float(self.deepl_input.text()), float(self.deepl_output.text()))
        cost_tracker.set_cost_per_million('fireworks', float(self.fireworks_input.text()), float(self.fireworks_output.text()))
        cost_tracker.set_cost_per_million('groq', float(self.groq_input.text()), float(self.groq_output.text()))
        cost_tracker.set_cost_per_million('together', float(self.together_input.text()), float(self.together_output.text()))
        cost_tracker.set_cost_per_million('ollama', float(self.ollama_input.text()), float(self.ollama_output.text()))

        # 11. Initialize with default English translation if no settings loaded or just to be safe
        # Получаем код языка, который мы только что загрузили или дефолтный
        current_lang_code = self.cmb_ui_lang.currentData() or 'english'
        self._apply_translations(current_lang_code)

        # 12. Check for updates in background
        self._check_updates_async()
    
    # --- UI TRANSLATION LOGIC ---

    def _apply_translations(self, lang_code: str):
        super()._apply_translations(lang_code)

        # 4. Review Interface
        if hasattr(self.review_interface, '_apply_translations'):
             self.review_interface._apply_translations(lang_code)
        else:
             self._retranslate_widgets(self.review_interface, lang_code)

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
            'rpm_limit': self.spn_rpm.value(),
            'glossary': self.ed_glossary.text(),
            'cache': self.ed_cache.text(),
            'cache_type': self.cmb_cache_type.currentText(),
            'reuse_prev_loc': self.chk_reuse_prev.isChecked(),
            'mark_loc': self.chk_mark_loc.isChecked(),
            'use_mod_name': self.chk_use_mod_name.isChecked(),
            'mod_name': self.ed_mod_name.text(),
            'batch_translation': self.chk_batch_mode.isChecked(),
            'chunk_size': self.spn_chunk_size.value(),
            'g4f_model': self.ed_g4f_model.text(),
            'g4f_api_key': self.ed_g4f_api_key.text(),
            'g4f_async': self.chk_g4f_async.isChecked(),
            'g4f_cc': self.spn_g4f_cc.value(),
            'io_api_key': self.ed_io_api_key.text(),
            'io_base_url': self.ed_io_base.text(),
            'io_async': self.chk_io_async.isChecked(),
            'io_model': self.cmb_io_model.currentText(),
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
            self.spn_rpm.setValue(int(settings.get('rpm_limit', 60)))

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

            # Additional settings
            self.chk_use_mod_name.setChecked(bool(settings.get('use_mod_name', False)))
            self.ed_mod_name.setText(settings.get('mod_name', ''))

            # Update inplace UI state
            self._toggle_inplace()

        except Exception as e:
            log_manager.error(f"Failed to load settings: {e}")
            return False

        return True

    def _load_env_defaults(self):
        """Load default values from .env file if not set in settings."""
        super()._load_env_defaults()

        # Additional API keys
        if not self.ed_yandex_translate_api_key.text().strip():
            api_key = get_api_key('yandex_translate')
            if api_key:
                self.ed_yandex_translate_api_key.setText(api_key)

        if not self.ed_yandex_cloud_api_key.text().strip():
            api_key = get_api_key('yandex_cloud')
            if api_key:
                self.ed_yandex_cloud_api_key.setText(api_key)

        if not self.ed_deepl_api_key.text().strip():
            api_key = get_api_key('deepl')
            if api_key:
                self.ed_deepl_api_key.setText(api_key)

        if not self.ed_fireworks_api_key.text().strip():
            api_key = get_api_key('fireworks')
            if api_key:
                self.ed_fireworks_api_key.setText(api_key)

        if not self.ed_groq_api_key.text().strip():
            api_key = get_api_key('groq')
            if api_key:
                self.ed_groq_api_key.setText(api_key)

        if not self.ed_together_api_key.text().strip():
            api_key = get_api_key('together')
            if api_key:
                self.ed_together_api_key.setText(api_key)

        # Set cost currency
        currency = get_cost_currency()
        if currency:
            cost_tracker.set_currency(currency)

    # UI components and navigation are initialized in ui_interfaces.py

    # --- LOGIC METHODS ---

    def _show_about(self):
        w = AboutDialog(self)
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
            p = os.path.join(out, ".hoi4loc_cache.db") if out else ""
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
            "rpm_limit": self.spn_rpm.value(),
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
            "yandex_translate_api_key": self.ed_yandex_translate_api_key.text(),
            "yandex_iam_token": self.ed_yandex_iam_token.text(),
            "yandex_folder_id": self.ed_yandex_folder_id.text(),
            "yandex_cloud_api_key": self.ed_yandex_cloud_api_key.text(),
            "yandex_cloud_model": self.ed_yandex_cloud_model.text(),
            "yandex_async": self.chk_yandex_async.isChecked(),
            "yandex_cc": self.spn_yandex_cc.value(),
            "deepl_api_key": self.ed_deepl_api_key.text(),
            "fireworks_api_key": self.ed_fireworks_api_key.text(),
            "fireworks_model": self.ed_fireworks_model.text(),
            "fireworks_async": self.chk_fireworks_async.isChecked(),
            "fireworks_cc": self.spn_fireworks_cc.value(),
            "groq_api_key": self.ed_groq_api_key.text(),
            "groq_model": self.ed_groq_model.text(),
            "groq_async": self.chk_groq_async.isChecked(),
            "groq_cc": self.spn_groq_cc.value(),
            "together_api_key": self.ed_together_api_key.text(),
            "together_model": self.ed_together_model.text(),
            "together_async": self.chk_together_async.isChecked(),
            "together_cc": self.spn_together_cc.value(),
            "ollama_model": self.ed_ollama_model.text(),
            "ollama_base_url": self.ed_ollama_base_url.text(),
            "ollama_async": self.chk_ollama_async.isChecked(),
            "ollama_cc": self.spn_ollama_cc.value(),
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
            self.spn_rpm.setValue(int(data.get("rpm_limit", 60)))
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
            self.ed_yandex_translate_api_key.setText(data.get("yandex_translate_api_key",""))
            self.ed_yandex_iam_token.setText(data.get("yandex_iam_token",""))
            self.ed_yandex_folder_id.setText(data.get("yandex_folder_id",""))
            self.ed_yandex_cloud_api_key.setText(data.get("yandex_cloud_api_key",""))
            self.ed_yandex_cloud_model.setText(data.get("yandex_cloud_model",""))
            self.chk_yandex_async.setChecked(bool(data.get("yandex_async", True)))
            self.spn_yandex_cc.setValue(int(data.get("yandex_cc", 6)))
            self.ed_deepl_api_key.setText(data.get("deepl_api_key",""))
            self.ed_fireworks_api_key.setText(data.get("fireworks_api_key",""))
            self.ed_fireworks_model.setText(data.get("fireworks_model",""))
            self.chk_fireworks_async.setChecked(bool(data.get("fireworks_async", True)))
            self.spn_fireworks_cc.setValue(int(data.get("fireworks_cc", 6)))
            self.ed_groq_api_key.setText(data.get("groq_api_key",""))
            self.ed_groq_model.setText(data.get("groq_model",""))
            self.chk_groq_async.setChecked(bool(data.get("groq_async", True)))
            self.spn_groq_cc.setValue(int(data.get("groq_cc", 6)))
            self.ed_together_api_key.setText(data.get("together_api_key",""))
            self.ed_together_model.setText(data.get("together_model",""))
            self.chk_together_async.setChecked(bool(data.get("together_async", True)))
            self.spn_together_cc.setValue(int(data.get("together_cc", 6)))
            self.ed_ollama_model.setText(data.get("ollama_model",""))
            self.ed_ollama_base_url.setText(data.get("ollama_base_url",""))
            self.chk_ollama_async.setChecked(bool(data.get("ollama_async", True)))
            self.spn_ollama_cc.setValue(int(data.get("ollama_cc", 6)))

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
            sqlite_cache_extension=load_settings().get('sqlite_cache_extension', '.db'),
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
        out = validated_settings.get('out', settings.get("out") or settings.get("src"))
        in_place = self.chk_inplace.isChecked()
        
        cache_path = (self.ed_cache.text().strip() or None)
        if cache_path is None:
            base = out or src
            cache_path = os.path.join(base, ".hoi4loc_cache.db")
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
            rpm_limit=self.spn_rpm.value(),
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
            use_mod_name=self.chk_use_mod_name.isChecked(),
            mod_name=self.ed_mod_name.text().strip() or None,
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
        from ..utils.fs import collect_localisation_files
        self._total_files = len(collect_localisation_files(cfg.src_dir))
        self.lbl_stats.setText(f"Words: 0 | Keys: 0 | Files: 0/{self._total_files}")
        self._translation_cost_start = cost_tracker._session_cost
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
        # Calculate cost spent on this translation
        spent = cost_tracker._session_cost - self._translation_cost_start
        currency = cost_tracker._default_currency
        if spent == 0:
            cost_text = "free"
        else:
            cost_text = f"{spent:.4f} {currency}"
        InfoBar.success("Finished", f"Translation completed successfully. Cost: {cost_text}", parent=self, duration=5000)
        self._append_log(f"Translation cost: {cost_text}")
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

            self.review_interface.load_data(file_path, data, files)
            QApplication.processEvents()
            self._append_log(f"Loaded {len(data)} entries for review")
                
        except Exception as e:
            self._append_log(f"Error loading review data: {e}")

    def _on_review_save(self, data: list):
        try:
            file_path = self.review_interface.current_file_path
            if not file_path:
                InfoBar.warning("Save Error", "No file loaded for saving", parent=self)
                return

            dst_lang = self.cmb_dst_lang.currentText()
            from ..parsers.paradox_yaml import save_yaml_file
            save_yaml_file(file_path, data, dst_lang)

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

            if self._retranslate_worker is not None:
                InfoBar.warning("Retranslate", "Retranslation already in progress", parent=self)
                return

            self._append_log(f"Retranslating {len(selected_items)} selected items")
            InfoBar.success("Retranslate", f"Retranslating {len(selected_items)} items", parent=self)

            # Create config for retranslation
            cfg = JobConfig(
                src_dir=self.ed_src.text().strip(),
                out_dir=self.ed_out.text().strip() if not self.chk_inplace.isChecked() else self.ed_src.text().strip(),
                src_lang=self.cmb_src_lang.currentText(),
                dst_lang=self.cmb_dst_lang.currentText(),
                model_key=self.cmb_model.currentText(),
                temperature=self.spn_temp.value() / 100.0,
                in_place=self.chk_inplace.isChecked(),
                skip_existing=self.chk_skip_exist.isChecked(),
                strip_md=self.chk_strip_md.isChecked(),
                batch_size=self.spn_batch.value(),
                rename_files=self.chk_rename_files.isChecked(),
                files_concurrency=self.spn_files_cc.value(),
                key_skip_regex=(self.ed_key_skip.text().strip() or None),
                cache_path=(self.ed_cache.text().strip() or None),
                glossary_path=(self.ed_glossary.text().strip() or None),
                prev_loc_dir=(self.ed_prev.text().strip() or None),
                reuse_prev_loc=self.chk_reuse_prev.isChecked(),
                mark_loc_flag=self.chk_mark_loc.isChecked(),
                batch_translation=self.chk_batch_mode.isChecked(),
                chunk_size=self.spn_chunk_size.value(),
                rpm_limit=self.spn_rpm.value(),
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
                use_mod_name=self.chk_use_mod_name.isChecked(),
                mod_name=self.ed_mod_name.text().strip() or None,
            )

            self._retranslate_worker = RetranslateWorker(cfg, selected_items)
            self._retranslate_worker.translation_done.connect(self._on_retranslate_done)
            self._retranslate_worker.log.connect(self._append_log)
            self._retranslate_worker.start()

        except Exception as e:
            InfoBar.error("Retranslate Error", str(e), parent=self)
            self._append_log(f"Error retranslating: {e}")

    def _on_retranslate_done(self, translations: list):
        self.review_interface.update_translations(translations)
        self._retranslate_worker = None
        InfoBar.success("Retranslate", f"Updated {len(translations)} translations", parent=self)

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
        else:
            InfoBar.info(
                title="No Updates",
                content="You are using the latest version.",
                parent=self
            )

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