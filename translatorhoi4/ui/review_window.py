# translatorhoi4/ui/review_window.py
import re
from typing import List, Dict, Optional
from collections import deque

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QHeaderView, 
    QTableWidgetItem, QFrame
)
from PyQt6.QtGui import QColor, QBrush, QKeySequence, QShortcut

from qfluentwidgets import (
    TableWidget, PrimaryPushButton, PushButton,
    FluentIcon as FIF, Action,
    RoundMenu, CommandBar, ToolButton,
    InfoBar, InfoBarPosition, SearchLineEdit,
    CheckBox
)

# Импортируем функцию перевода (убедитесь, что файл translations.py существует в той же папке ui)
from .translations import translate_text

class Hoi4Validator:
    """Helper class to validate HoI4 syntax."""
    
    # Regex patterns
    VAR_PATTERN = re.compile(r'\[.*?\]')      # [Root.GetName]
    COLOR_PATTERN = re.compile(r'§[a-zA-Z!]') # §Y, §!
    ICON_PATTERN = re.compile(r'£.*?£')       # £integrity_icon£
    QUOTE_PATTERN = re.compile(r'(?<!\\)"')   # Unescaped quotes

    @staticmethod
    def validate(original: str, translation: str) -> List[str]:
        errors = []
        
        # 1. Check Variables [...]
        orig_vars = set(Hoi4Validator.VAR_PATTERN.findall(original))
        trans_vars = set(Hoi4Validator.VAR_PATTERN.findall(translation))
        missing_vars = orig_vars - trans_vars
        if missing_vars:
            errors.append(f"Missing variables: {', '.join(missing_vars)}")

        # 2. Check Colors §...
        if len(Hoi4Validator.COLOR_PATTERN.findall(original)) != len(Hoi4Validator.COLOR_PATTERN.findall(translation)):
             errors.append("Mismatch in color codes (§)")

        # 3. Check Icons £...£
        orig_icons = set(Hoi4Validator.ICON_PATTERN.findall(original))
        trans_icons = set(Hoi4Validator.ICON_PATTERN.findall(translation))
        if orig_icons != trans_icons:
            errors.append("Mismatch in icons (£...£)")

        # 4. Check unescaped quotes
        if '"' in translation and '\\"' not in translation and '"' not in original:
             errors.append("Possible unescaped quote")

        return errors


class ReviewInterface(QWidget):
    """
    Main Interface for reviewing translations.
    """
    
    # Signals
    save_requested = pyqtSignal(list)
    retranslate_requested = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("reviewInterface")
        
        # Internal State
        self.current_file_path = None
        self.current_file_index = 0
        self.all_files = []
        self.all_data = []
        
        # Undo stack
        self._undo_stack = deque(maxlen=50)
        self._undo_shortcut = None
        
        self._setup_ui()
        self._setup_shortcuts()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(20, 10, 20, 0)
        
        self.lbl_filename = PushButton("No file loaded", self, FIF.DOCUMENT)
        self.lbl_filename.setEnabled(False)
        
        self.search_bar = SearchLineEdit()
        self.search_bar.setPlaceholderText("Search key or text...")
        self.search_bar.textChanged.connect(self._filter_table)
        self.search_bar.setFixedWidth(300)

        toolbar_layout.addWidget(self.lbl_filename)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.search_bar)
        
        layout.addLayout(toolbar_layout)

        # --- Table ---
        self.table = TableWidget(self)
        self.table.setColumnCount(5)
        # Исходные заголовки (на английском)
        self.table.setHorizontalHeaderLabels([
            "", 
            "Key", 
            "Original", 
            "Translation (Editable)", 
            "Status"
        ])
        
        self.table.verticalHeader().hide()
        self.table.setBorderRadius(8)
        self.table.setWordWrap(False)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 40)
        
        layout.addWidget(self.table)

        # --- Action Bar ---
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(20, 10, 20, 20)
        
        self.btn_prev = PushButton("Previous File", self, FIF.CARE_LEFT_SOLID)
        self.btn_prev.clicked.connect(self._on_prev_file)
        
        self.btn_next = PushButton("Next File", self, FIF.CARE_RIGHT_SOLID)
        self.btn_next.clicked.connect(self._on_next_file)
        
        self.btn_undo = PushButton("Undo (Ctrl+Z)", self, FIF.RETURN)
        self.btn_undo.clicked.connect(self._undo)
        
        self.btn_retranslate = PushButton("Retranslate Selected", self, FIF.SYNC)
        self.btn_retranslate.clicked.connect(self._on_retranslate)
        
        self.btn_save = PrimaryPushButton("Save File", self, FIF.SAVE)
        self.btn_save.clicked.connect(self._on_save)

        action_layout.addWidget(self.btn_prev)
        action_layout.addWidget(self.btn_next)
        action_layout.addStretch(1)
        action_layout.addWidget(self.btn_undo)
        action_layout.addWidget(self.btn_retranslate)
        action_layout.addWidget(self.btn_save)
        
        layout.addLayout(action_layout)
    
    def _setup_shortcuts(self):
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._undo)

    # --- TRANSLATION LOGIC (ADDED) ---

    def _apply_translations(self, lang_code: str):
        """
        Apply translations to all widgets in this interface.
        Called by MainWindow when language changes.
        """
        # 1. Translate all standard QWidgets (Buttons, Labels, SearchBar)
        for widget in self.findChildren(QWidget):
            self._translate_single_widget(widget, lang_code)
        
        # 2. Translate Table Headers (Special handling required)
        # We assume the column order is fixed: ["", "Key", "Original", "Translation", "Status"]
        # Empty string is for checkbox column, doesn't need translation
        headers = ["", "Key", "Original", "Translation (Editable)", "Status"]
        translated_headers = []
        for h in headers:
            if not h:
                translated_headers.append("")
            else:
                translated_headers.append(translate_text(h, lang_code))
        
        self.table.setHorizontalHeaderLabels(translated_headers)

    def _translate_single_widget(self, widget: QWidget, lang_code: str):
        """
        Helper to translate a single widget using cached original text.
        This mirrors the logic in MainWindow to prevent 'translating a translation'.
        """
        # Properties to check: (Property Name, Setter Function Name)
        properties_to_translate = [
            ("text", "setText"),
            ("placeholderText", "setPlaceholderText")
        ]

        for prop_name, setter_name in properties_to_translate:
            if not hasattr(widget, prop_name) or not hasattr(widget, setter_name):
                continue
            
            try:
                # Get current value (e.g. widget.text())
                getter = getattr(widget, prop_name)
                current_val = getter()
            except Exception:
                continue

            if not isinstance(current_val, str) or not current_val:
                continue
            
            # --- CACHING LOGIC ---
            # Cache key example: "original_text"
            cache_key = f"original_{prop_name}"
            original_text = widget.property(cache_key)

            if original_text is None:
                # First time: Save English/Original text
                original_text = current_val
                widget.setProperty(cache_key, original_text)
            
            # Always translate from ORIGINAL to TARGET
            translated_text = translate_text(original_text, lang_code)

            # Apply only if different
            if translated_text != current_val:
                setter = getattr(widget, setter_name)
                setter(translated_text)

    # --- END TRANSLATION LOGIC ---
    
    def _save_undo_state(self):
        if self.all_data:
            snapshot = [dict(row) for row in self.all_data]
            self._undo_stack.append(snapshot)
    
    def _undo(self):
        if not self._undo_stack:
            InfoBar.info("Undo", "Nothing to undo", parent=self, duration=1000)
            return
        
        self.all_data = self._undo_stack.pop()
        self._refresh_table_from_data()
        InfoBar.success("Undo", "Restored previous state", parent=self, duration=1000)
    
    def _refresh_table_from_data(self):
        if not self.all_data:
            return
        
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.all_data))
        
        for row_idx, item in enumerate(self.all_data):
            # 0. Checkbox
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_idx, 0, chk_item)
            
            # 1. Key
            key_item = QTableWidgetItem(item.get('key', ''))
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row_idx, 1, key_item)
            
            # 2. Original
            orig_text = item.get('original', '')
            orig_item = QTableWidgetItem(orig_text)
            orig_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            orig_item.setToolTip(orig_text)
            self.table.setItem(row_idx, 2, orig_item)
            
            # 3. Translation
            trans_text = item.get('translation', '')
            trans_item = QTableWidgetItem(trans_text)
            self.table.setItem(row_idx, 3, trans_item)
            
            # 4. Status
            self._validate_row(row_idx, orig_text, trans_text)
        
        self.table.blockSignals(False)

    def load_files(self, files: list):
        self.all_files = files
        self.current_file_index = 0
        if files:
            self._load_current_file()
        else:
            self.current_file_path = None
            self.lbl_filename.setText("No files loaded")
            self.all_data = []
            self.table.setRowCount(0)
        self._update_navigation_buttons()
    
    def _load_current_file(self):
        if not self.all_files or self.current_file_index < 0 or self.current_file_index >= len(self.all_files):
            return
        
        from ..parsers.paradox_yaml import parse_yaml_file
        file_path = self.all_files[self.current_file_index]
        # Обычно здесь должен быть код парсинга, но MainWindow передает данные через load_data.
        # Если load_files используется для навигации, нам нужно уметь парсить файлы заново:
        try:
            # Для простоты считаем, что MainWindow управляет первичной загрузкой,
            # а здесь мы просто обновляем путь.
            # Но если нужно переоткрыть файл:
            pass
        except Exception:
            pass
        
        # NOTE: В текущей архитектуре MainWindow вызывает load_data напрямую для первого файла.
        # Для кнопок Prev/Next нам нужно запросить данные у MainWindow или парсить самим.
        # Для упрощения оставим обновление UI, а логику загрузки данных лучше вынести в контроллер.
        # Но чтобы кнопки работали, сделаем заглушку на текст:
        self.lbl_filename.setText(f"File {self.current_file_index + 1}/{len(self.all_files)}: {file_path}")
    
    def _on_prev_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            # Real implementation would load data here
            self._update_navigation_buttons()
            # Signal to main window to load specific file? 
            # Or just parse here if we have parser imports.
            self._reload_current_file_content()
    
    def _on_next_file(self):
        if self.current_file_index < len(self.all_files) - 1:
            self.current_file_index += 1
            self._update_navigation_buttons()
            self._reload_current_file_content()

    def _reload_current_file_content(self):
        """Helper to reload data when navigating files."""
        if not self.all_files: return
        
        path = self.all_files[self.current_file_index]
        # Need to import parser here or pass it in
        from ..parsers.paradox_yaml import parse_source_and_translation
        # Note: We need source file path too for proper diff logic.
        # This part depends on how your MainWindow manages paths.
        # Assuming simple parse for now:
        from ..parsers.paradox_yaml import parse_yaml_file
        data = parse_yaml_file(path)
        # Simple data load (no diff with source available in this scope easily without more context)
        self.load_data(path, data)

    def _update_navigation_buttons(self):
        self.btn_prev.setEnabled(self.current_file_index > 0)
        self.btn_next.setEnabled(self.current_file_index < len(self.all_files) - 1)
    
    def load_data(self, file_path: str, data: List[Dict], all_files: List[str] = None):
        """
        Loads data into the table.
        all_files: optional list of all files to enable navigation
        """
        self.current_file_path = file_path
        if all_files:
            self.all_files = all_files
            if file_path in all_files:
                self.current_file_index = all_files.index(file_path)
            self._update_navigation_buttons()
        
        self.lbl_filename.setText(f"File: {file_path}")
        self.all_data = data
        self.table.setRowCount(0)
        
        self.table.blockSignals(True)
        self.table.setRowCount(len(data))
        
        for row_idx, item in enumerate(data):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_idx, 0, chk_item)
            
            key_item = QTableWidgetItem(item.get('key', ''))
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row_idx, 1, key_item)
            
            orig_text = item.get('original', '')
            orig_item = QTableWidgetItem(orig_text)
            orig_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            orig_item.setToolTip(orig_text)
            self.table.setItem(row_idx, 2, orig_item)
            
            trans_text = item.get('translation', '')
            trans_item = QTableWidgetItem(trans_text)
            self.table.setItem(row_idx, 3, trans_item)
            
            self._validate_row(row_idx, orig_text, trans_text)

        self.table.blockSignals(False)
        self.table.itemChanged.connect(self._on_item_changed)

    def _validate_row(self, row: int, original: str, translation: str):
        errors = Hoi4Validator.validate(original, translation)
        
        status_item = QTableWidgetItem()
        trans_item = self.table.item(row, 3)
        
        if errors:
            status_item.setText("⚠️ " + errors[0])
            status_item.setForeground(QBrush(QColor("#ff4d4f")))
            status_item.setToolTip("\n".join(errors))
            trans_item.setBackground(QBrush(QColor(60, 20, 20)))
        else:
            status_item.setText("OK")
            status_item.setForeground(QBrush(QColor("#52c41a")))
            trans_item.setBackground(QBrush(Qt.GlobalColor.transparent))

        status_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, 4, status_item)

    def _on_item_changed(self, item: QTableWidgetItem):
        if item.column() == 3:
            row = item.row()
            original = self.table.item(row, 2).text()
            translation = item.text()
            
            self._save_undo_state()
            
            if row < len(self.all_data):
                self.all_data[row]['translation'] = translation
                
            self.table.blockSignals(True)
            self._validate_row(row, original, translation)
            self.table.blockSignals(False)

    def _on_save(self):
        if not self.current_file_path:
            return
        self.save_requested.emit(self.all_data)
        InfoBar.success(
            title="Saved",
            content=f"Changes for {self.current_file_path} ready to write.",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=2000
        )

    def _on_retranslate(self):
        selected_indices = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item.checkState() == Qt.CheckState.Checked:
                key = self.table.item(row, 1).text()
                original = self.table.item(row, 2).text()
                selected_indices.append({'key': key, 'original': original, 'row': row})
        
        if not selected_indices:
            InfoBar.warning("Selection", "No rows selected for re-translation", parent=self)
            return

        self.retranslate_requested.emit(selected_indices)

    def _filter_table(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 1).text().lower()
            orig = self.table.item(row, 2).text().lower()
            trans = self.table.item(row, 3).text().lower()
            
            is_visible = (text in key) or (text in orig) or (text in trans)
            self.table.setRowHidden(row, not is_visible)