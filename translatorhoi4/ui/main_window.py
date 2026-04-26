"""Main application window - extends base UI with cost tracking, review, and system tray."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from qfluentwidgets import (
    InfoBar, InfoBarPosition, MessageBox
)

from ..utils.env import get_api_key, get_cost_currency
from ..utils.logging_config import setup_logging, log_manager
from ..translator.cost import cost_tracker
from ..translator.engine import TranslateWorker, RetranslateWorker
from .ui_interfaces import MainWindow as BaseMainWindow
from .about import AboutDialog


class MainWindow(BaseMainWindow):
    """Extended main window with cost tracking, review interface, pause, and system tray."""

    def __init__(self):
        super().__init__()

        # Initialize additional workers
        self._retranslate_worker: Optional[RetranslateWorker] = None
        self._force_quit = False  # Flag to distinguish between hide and exit

        # Setup logging
        setup_logging()

        # Initialize cost tracker
        cost_tracker.start_session()

        # Set cost configuration from UI
        self._update_cost_tracker()

        # Apply logic hooks
        self._switch_backend_settings(self.cmb_model.currentText())

        # Setup UI language change handler
        self.cmb_ui_lang.currentIndexChanged.connect(self._on_ui_lang_changed)

        # Load saved settings
        self._load_settings()

        # Load API keys from .env as defaults
        self._load_env_defaults()

        # Apply translations
        current_lang_code = self.cmb_ui_lang.currentData() or 'english'
        self._apply_translations(current_lang_code)

        # Check for updates
        self._latest_update_info = {}
        self._check_updates_async()

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._save_settings)

    def _update_cost_tracker(self):
        """Update cost tracker with current UI values."""
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
        cost_tracker.set_cost_per_million('mistral', float(self.mistral_input.text()), float(self.mistral_output.text()))
        cost_tracker.set_cost_per_million('nvidia', float(self.nvidia_input.text()), float(self.nvidia_output.text()))

    # --- UI TRANSLATION LOGIC ---

    def _apply_translations(self, lang_code: str):
        super()._apply_translations(lang_code)

        # Translate Review Interface
        if hasattr(self.review_interface, '_apply_translations'):
            self.review_interface._apply_translations(lang_code)
        else:
            self._retranslate_widgets(self.review_interface, lang_code)

    # --- SETTINGS ---

    def _save_settings(self):
        """Save current settings to cache file."""
        super()._save_settings()

    def _load_settings(self):
        """Load settings from cache file."""
        result = super()._load_settings()
        if result:
            # Update cost tracker after loading settings
            self._update_cost_tracker()
        return result

    def _load_env_defaults(self):
        """Load default values from .env file if not set in settings."""
        super()._load_env_defaults()

        # Additional API keys
        for env_key, widget_attr in [
            ('yandex_translate', 'ed_yandex_translate_api_key'),
            ('yandex_cloud', 'ed_yandex_cloud_api_key'),
            ('deepl', 'ed_deepl_api_key'),
            ('fireworks', 'ed_fireworks_api_key'),
            ('groq', 'ed_groq_api_key'),
            ('together', 'ed_together_api_key'),
            ('mistral', 'ed_mistral_api_key'),
            ('nvidia', 'ed_nvidia_api_key'),
        ]:
            widget = getattr(self, widget_attr, None)
            if widget and not widget.text().strip():
                api_key = get_api_key(env_key)
                if api_key:
                    widget.setText(api_key)

    # --- ACTIONS ---

    def _show_about(self):
        """Show about dialog."""
        w = AboutDialog(self, update_info=self._latest_update_info)
        w.exec()

    def _start(self):
        """Start translation with cost tracking."""
        # Update cost tracker before starting
        self._update_cost_tracker()
        
        # Call parent implementation
        super()._start()

    def _on_done(self):
        """Handle translation completion with cost display."""
        super()._on_done()
        
        # Show cost summary
        total_cost = cost_tracker.get_total_cost()
        if total_cost > 0:
            currency = cost_tracker.get_currency()
            InfoBar.success(
                title=self.tr("Translation Complete"),
                content=self.tr(f"Total cost: {total_cost:.4f} {currency}"),
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM,
                duration=5000,
                parent=self
            )

    def _toggle_pause(self):
        """Toggle pause/resume translation."""
        if self._worker:
            if self._worker.is_paused():
                self._worker.resume()
                self.btn_pause.setText(self.tr("Pause"))
                self.act_pause.setText(self.tr("Pause"))
                self._append_log(self.tr("Translation resumed."))
                self._update_tray_status(status="Translating")
            else:
                self._worker.pause()
                self.btn_pause.setText(self.tr("Resume"))
                self.act_pause.setText(self.tr("Resume"))
                self._append_log(self.tr("Translation paused."))
                self._update_tray_status(status="Paused")

    def _on_update_check_finished(self, update_info):
        self._latest_update_info = update_info or {}
        super()._on_update_check_finished(update_info)

    # --- REVIEW INTERFACE ---

    def _on_review_save(self, data: list):
        """Save reviewed translation."""
        try:
            file_path = self.review_interface.current_file_path
            if not file_path:
                InfoBar.warning("Save Error", "No file loaded for saving", parent=self)
                return

            # Import the save function
            from ..parsers.paradox_yaml import save_yaml_file
            
            # Determine the target language from the current destination language setting
            dst_lang = self.cmb_dst_lang.currentText()
            
            # Save the file with the reviewed data
            save_yaml_file(file_path, data, dst_lang)
            
            InfoBar.success(
                title=self.tr("Saved"),
                content=self.tr(f"Changes saved to {os.path.basename(file_path)}"),
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM,
                duration=2000,
                parent=self
            )
            self._append_log(f"Saved changes to {file_path}")
        except Exception as e:
            InfoBar.error(
                title=self.tr("Save Error"),
                content=str(e),
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM,
                duration=-1,
                parent=self
            )

    def _on_review_retranslate(self, selected_items: list):
        """Retranslate selected items from review."""
        if self._translating:
            InfoBar.warning(
                title=self.tr("Busy"),
                content=self.tr("Please wait for current translation to finish"),
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM,
                duration=3000,
                parent=self
            )
            return

        if not selected_items:
            InfoBar.warning(
                title=self.tr("Retranslate"),
                content=self.tr("No items selected for retranslation"),
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM,
                duration=3000,
                parent=self
            )
            return

        from ..translator.engine import JobConfig

        file_path = self.review_interface.current_file_path
        if not file_path:
            InfoBar.warning("Retranslate Error", "No file loaded", parent=self)
            return

        cfg = JobConfig(
            src_dir=os.path.dirname(file_path),
            out_dir=os.path.dirname(file_path),
            prev_dir=self.ed_prev.text().strip() or None,
            src_lang=self.cmb_src_lang.currentText(),
            dst_lang=self.cmb_dst_lang.currentText(),
            model_key=self.cmb_model.currentText(),
            temperature=self.spn_temp.value() / 100.0,
            strip_md=self.chk_strip_md.isChecked(),
            rename_files=self.chk_rename_files.isChecked(),
            skip_existing=False,
            reuse_prev_loc=self.chk_reuse_prev.isChecked(),
            mark_loc=self.chk_mark_loc.isChecked(),
            key_skip_regex=self.ed_key_skip.text().strip() or None,
            batch_size=self.spn_batch.value(),
            files_cc=self.spn_files_cc.value(),
            rpm_limit=self.spn_rpm.value(),
            glossary_path=self.ed_glossary.text().strip() or None,
            cache_path=self.ed_cache.text().strip() or None,
            cache_type=self.cmb_cache_type.currentText(),
            batch_mode=self.chk_batch_mode.isChecked(),
            chunk_size=self.spn_chunk_size.value(),
            include_replace=self.chk_include_replace.isChecked(),
            specific_file=file_path,
        )

        self._retranslate_worker = RetranslateWorker(cfg, selected_items)
        self._retranslate_worker.progress.connect(self._on_progress)
        self._retranslate_worker.log.connect(self._append_log)
        self._retranslate_worker.translation_done.connect(self._on_retranslate_done)
        self._retranslate_worker.start()

        self._translating = True
        self.btn_go.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._update_tray_status(status="Retranslating", file_text=f"Retranslating {len(selected_items)} entries")
        self._append_log(self.tr(f"Retranslating {len(selected_items)} items from {os.path.basename(file_path)}"))

    def _on_retranslate_done(self):
        """Handle retranslation completion."""
        self._translating = False
        self.btn_go.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self._update_tray_status(status="Completed", file_text="Retranslation completed")
        InfoBar.success(
            title=self.tr("Done"),
            content=self.tr("Retranslation completed"),
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM,
            duration=3000,
            parent=self
        )

    # --- SYSTEM TRAY ---

    def closeEvent(self, event):
        """Handle window close event - minimize to tray or quit."""
        self._save_settings()
        if self._force_quit:
            super().closeEvent(event)
            return
        
        # Minimize to tray instead of closing
        event.ignore()
        self.hide()
        
        # Show tray notification
        if self.tray_icon:
            self.tray_icon.showMessage(
                self.tr("TranslatorHoi4"),
                self.tr("Application is running in system tray"),
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
