"""UI themes."""
DARK_QSS = """
* { color: #ECECEC; font-size: 13px; }
QMainWindow { background-color: #0F1113; }
QGroupBox {
  border: 1px solid #2A2F36; border-radius: 10px; margin-top: 10px;
  font-weight: 600; color: #E6E6E6;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QLabel { color: #E8E8E8; }
QLineEdit, QTextEdit, QComboBox, QSpinBox {
  background: #151A1F; border: 1px solid #2E3640; border-radius: 8px; padding: 6px 8px;
  color: #F0F0F0; selection-background-color: #3478F6; selection-color: #FFFFFF;
}
QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
  color: #9AA0A6; background: #161A1F; border-color: #2A3036;
}
QLineEdit[echoMode="2"] { letter-spacing: 0.2em; }
QComboBox QAbstractItemView {
  background: #151A1F; color: #F0F0F0; selection-background-color: #3478F6; selection-color: #FFFFFF;
  border: 1px solid #2E3640;
}
QProgressBar {
  background: #13171B; border: 1px solid #2E3640; border-radius: 8px; text-align: center;
}
QProgressBar::chunk { background-color: #3478F6; border-radius: 8px; }
QPushButton {
  background: #1B2128; border: 1px solid #2F3742; border-radius: 10px; padding: 8px 12px;
}
QPushButton:hover { background: #222A33; }
QPushButton:pressed { background: #171C22; }
QMenuBar { background: #0F1113; color: #EDEDED; }
QMenu { background: #151A1F; border: 1px solid #2E3640; }
QCheckBox { spacing: 6px; }
QTabWidget::pane { border: 1px solid #2A2F36; }
QTabBar::tab { background: #151A1F; padding: 6px 10px; border: 1px solid #2A2F36; border-bottom: none; }
QTabBar::tab:selected { background: #1B2128; }
"""
