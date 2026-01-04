# translatorhoi4/ui/review_window.py
import re
from typing import List, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QHeaderView, 
    QTableWidgetItem, QFrame
)
from PyQt6.QtGui import QColor, QBrush

from qfluentwidgets import (
    TableWidget, PrimaryPushButton, PushButton,
    FluentIcon as FIF, Action,
    RoundMenu, CommandBar, ToolButton,
    InfoBar, InfoBarPosition, SearchLineEdit,
    CheckBox
)
# from qfluentwidgets import SubInterface  <-- УДАЛЕНО: Этот класс вызывает ошибку

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
        # We just check counts roughly, as order might change slightly, 
        # but usually count should match.
        if len(Hoi4Validator.COLOR_PATTERN.findall(original)) != len(Hoi4Validator.COLOR_PATTERN.findall(translation)):
             errors.append("Mismatch in color codes (§)")

        # 3. Check Icons £...£
        orig_icons = set(Hoi4Validator.ICON_PATTERN.findall(original))
        trans_icons = set(Hoi4Validator.ICON_PATTERN.findall(translation))
        if orig_icons != trans_icons:
            errors.append("Mismatch in icons (£...£)")

        # 4. Check unescaped quotes (common yaml breaker)
        # If original has no unescaped quotes but translation does -> suspicious
        # (Simplified logic)
        if '"' in translation and '\\"' not in translation and '"' not in original:
             errors.append("Possible unescaped quote")

        return errors

# ИЗМЕНЕНО: Наследуемся от QWidget вместо SubInterface
class ReviewInterface(QWidget):
    """
    Main Interface for reviewing translations.
    This acts as the 'Page' inside the MainWindow.
    """
    
    # Signals to communicate with MainWindow or Controller
    save_requested = pyqtSignal(list) # Emits list of modified file data
    retranslate_requested = pyqtSignal(list) # Emits list of (key, original) to re-queue
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("reviewInterface")
        
        # Internal State
        self.current_file_path = None
        self.all_data = [] # List of dicts representing rows
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(20, 10, 20, 0)
        
        self.lbl_filename = PushButton("No file loaded", self, FIF.DOCUMENT)
        self.lbl_filename.setEnabled(False) # Just used as a label with icon
        
        self.search_bar = SearchLineEdit()
        self.search_bar.setPlaceholderText("Search key or text...")
        self.search_bar.textChanged.connect(self._filter_table)
        self.search_bar.setFixedWidth(300)

        toolbar_layout.addWidget(self.lbl_filename)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.search_bar)
        
        layout.addLayout(toolbar_layout)

        # --- Table ---
        # Using qfluentwidgets.TableWidget
        self.table = TableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "", # Checkbox column
            "Key", 
            "Original", 
            "Translation (Editable)", 
            "Status"
        ])
        
        # Column setup
        self.table.verticalHeader().hide()
        self.table.setBorderRadius(8)
        self.table.setWordWrap(False) # HoI4 text can be long, maybe enable later
        
        # Resizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 40)
        
        layout.addWidget(self.table)

        # --- Action Bar (Bottom) ---
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(20, 10, 20, 20)
        
        self.btn_prev = PushButton("Previous File", self, FIF.CARE_LEFT_SOLID)
        self.btn_next = PushButton("Next File", self, FIF.CARE_RIGHT_SOLID)
        
        self.btn_retranslate = PushButton("Retranslate Selected", self, FIF.SYNC)
        self.btn_retranslate.clicked.connect(self._on_retranslate)
        
        self.btn_save = PrimaryPushButton("Save File", self, FIF.SAVE)
        self.btn_save.clicked.connect(self._on_save)

        action_layout.addWidget(self.btn_prev)
        action_layout.addWidget(self.btn_next)
        action_layout.addStretch(1)
        action_layout.addWidget(self.btn_retranslate)
        action_layout.addWidget(self.btn_save)
        
        layout.addLayout(action_layout)

    def load_data(self, file_path: str, data: List[Dict]):
        """
        Loads data into the table.
        data format: [{'key': str, 'original': str, 'translation': str}, ...]
        """
        self.current_file_path = file_path
        self.lbl_filename.setText(f"File: {file_path}")
        self.all_data = data
        self.table.setRowCount(0)
        
        # Optimization: Turn off signals during bulk load
        self.table.blockSignals(True)
        self.table.setRowCount(len(data))
        
        for row_idx, item in enumerate(data):
            # 0. Checkbox (Implemented manually via cell widget or simple checkable item)
            # TableWidget handles Checkable Items well
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_idx, 0, chk_item)
            
            # 1. Key
            key_item = QTableWidgetItem(item.get('key', ''))
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable) # Read-only
            self.table.setItem(row_idx, 1, key_item)
            
            # 2. Original
            orig_text = item.get('original', '')
            orig_item = QTableWidgetItem(orig_text)
            orig_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            orig_item.setToolTip(orig_text)
            self.table.setItem(row_idx, 2, orig_item)
            
            # 3. Translation (Editable)
            trans_text = item.get('translation', '')
            trans_item = QTableWidgetItem(trans_text)
            self.table.setItem(row_idx, 3, trans_item)
            
            # 4. Status / Validation
            self._validate_row(row_idx, orig_text, trans_text)

        self.table.blockSignals(False)
        
        # Connect change signal to validation
        self.table.itemChanged.connect(self._on_item_changed)

    def _validate_row(self, row: int, original: str, translation: str):
        """Runs validation logic and colors the row if needed."""
        errors = Hoi4Validator.validate(original, translation)
        
        status_item = QTableWidgetItem()
        trans_item = self.table.item(row, 3)
        
        if errors:
            status_item.setText("⚠️ " + errors[0])
            status_item.setForeground(QBrush(QColor("#ff4d4f"))) # Red text
            status_item.setToolTip("\n".join(errors))
            
            # Highlight translation cell background lightly
            trans_item.setBackground(QBrush(QColor(60, 20, 20))) # Dark red bg
        else:
            status_item.setText("OK")
            status_item.setForeground(QBrush(QColor("#52c41a"))) # Green text
            trans_item.setBackground(QBrush(Qt.GlobalColor.transparent))

        status_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, 4, status_item)

    def _on_item_changed(self, item: QTableWidgetItem):
        """Handle edits to the translation column."""
        if item.column() == 3: # Translation column
            row = item.row()
            original = self.table.item(row, 2).text()
            translation = item.text()
            
            # Update internal data
            if row < len(self.all_data):
                self.all_data[row]['translation'] = translation
                
            # Re-validate
            self.table.blockSignals(True) # Prevent recursion
            self._validate_row(row, original, translation)
            self.table.blockSignals(False)

    def _on_save(self):
        """Emits current data to be saved to disk."""
        if not self.current_file_path:
            return
        
        # Here you would typically reconstruct the file content
        # For now, we emit the data and let the Controller handle file writing
        self.save_requested.emit(self.all_data)
        
        InfoBar.success(
            title="Saved",
            content=f"Changes for {self.current_file_path} ready to write.",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=2000
        )

    def _on_retranslate(self):
        """Gathers checked rows and requests re-translation."""
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
        """Simple filter logic."""
        text = text.lower()
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 1).text().lower()
            orig = self.table.item(row, 2).text().lower()
            trans = self.table.item(row, 3).text().lower()
            
            is_visible = (text in key) or (text in orig) or (text in trans)
            self.table.setRowHidden(row, not is_visible)