"""Provider selector with beautiful UI and filtering."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame, QGraphicsDropShadowEffect
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit,
    FluentIcon as FIF, CardWidget, StrongBodyLabel, BodyLabel
)


def _get_resource_path(rel_path: str) -> str:
    """Get resource path for both development and PyInstaller builds."""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    else:
        # Development mode - use current directory
        base_path = Path(__file__).resolve().parent.parent.parent
    
    result = base_path / rel_path
    return str(result)

class ProviderCard(CardWidget):
    """Individual provider card with icon and info."""
    
    clicked = pyqtSignal(str)  # provider_key
    
    def __init__(self, provider_key: str, provider_info: dict, parent=None):
        super().__init__(parent)
        self.provider_key = provider_key
        self.provider_info = provider_info
        
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Icon
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(48, 48)
        self.icon_label.setScaledContents(True)
        
        # Try to load icon, fallback to text if not found
        icon_path = provider_info.get('icon', '')
        if icon_path:
            full_icon_path = _get_resource_path(icon_path)
            if os.path.exists(full_icon_path):
                self.icon_label.setPixmap(QPixmap(full_icon_path))
            else:
                # Try alternative paths for PyInstaller
                alt_paths = [
                    _get_resource_path(f"_internal/{icon_path}"),
                    _get_resource_path(f"{icon_path}"),
                    _get_resource_path(f"assets/providers/{Path(icon_path).name}")
                ]
                
                icon_loaded = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.icon_label.setPixmap(QPixmap(alt_path))
                        icon_loaded = True
                        break
                
                if not icon_loaded:
                    self._set_text_icon()
        else:
            self._set_text_icon()
        
        # Info section
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(5)
        
        # Provider name with better styling
        name_label = StrongBodyLabel(provider_info.get('display_name', self.provider_key))
        name_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 4px;
        """)
        info_layout.addWidget(name_label)
        
        # Description with better readability
        desc_label = BodyLabel(provider_info.get('description', ''))
        desc_label.setStyleSheet("""
            color: #cccccc;
            font-size: 13px;
            line-height: 1.4;
        """)
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)
        
        # Tags with better styling and colors
        if provider_info.get('tags'):
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(8)
            for tag in provider_info['tags']:
                tag_label = QLabel(tag)
                # Better tag colors based on category
                category = provider_info.get('category', 'premium')
                tag_colors = {
                    'free': 'background-color: #2d5a2d; color: #a0ffa0;',
                    'affordable': 'background-color: #2d4a6b; color: #a0d0ff;',
                    'premium': 'background-color: #5a4a2d; color: #ffd0a0;',
                    'local': 'background-color: #4a4a4a; color: #d0d0d0;'
                }
                tag_style = tag_colors.get(category, 'background-color: #4a4a4a; color: #ffffff;')
                
                tag_label.setStyleSheet(f"""
                    QLabel {{
                        {tag_style}
                        padding: 4px 10px;
                        border-radius: 15px;
                        font-size: 11px;
                        font-weight: bold;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                    }}
                """)
                tags_layout.addWidget(tag_label)
            tags_layout.addStretch()
            info_layout.addLayout(tags_layout)
        
        # Select button with better styling
        select_btn = PrimaryPushButton("Select")
        select_btn.setFixedWidth(90)
        select_btn.setStyleSheet("""
            PrimaryPushButton {
                background-color: #4a90e2;
                border: 1px solid #357abd;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
                font-size: 13px;
            }
            PrimaryPushButton:hover {
                background-color: #5ba0f2;
                border-color: #4a90e2;
            }
            PrimaryPushButton:pressed {
                background-color: #357abd;
            }
        """)
        select_btn.clicked.connect(self._on_clicked)
        
        layout.addWidget(self.icon_label)
        layout.addWidget(info_widget, 1)
        layout.addWidget(select_btn)
        
        # Set card style - more opaque and better contrast
        self.setStyleSheet("""
            ProviderCard {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #555555;
                border-radius: 12px;
                margin: 8px;
                padding: 5px;
            }
            ProviderCard:hover {
                background-color: rgba(55, 55, 55, 0.98);
                border-color: #777777;
                border-width: 2px;
            }
        """)
    
    def _set_text_icon(self, bg_color: str = None, short_name: str = None):
        """Set text-based icon as fallback with better styling."""
        if not bg_color:
            colors = {
                'free': '#4CAF50',      # Green
                'affordable': '#2196F3', # Blue
                'premium': '#FF9800',    # Orange
                'local': '#757575'       # Gray
            }
            bg_color = colors.get(self.provider_info.get('category', 'premium'), '#FF9800')
        
        if not short_name:
            short_name = self.provider_info.get('short_name', self.provider_key[0])
        
        # Determine text color based on background brightness
        import colorsys
        rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        text_color = '#000000' if hsv[2] > 0.5 else '#ffffff'
        
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                border-radius: 10px;
                font-size: 22px;
                color: {text_color};
                font-weight: bold;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }}
        """)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setText(short_name)
        
    
    def _on_clicked(self):
        self.clicked.emit(self.provider_key)


class ProviderSelectorDialog(QWidget):
    """Beautiful provider selector dialog."""
    
    provider_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select AI Provider")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.resize(650, 550)
        # Set window to be more opaque and centered
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        
        # Add drop shadow effect for modern look
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Provider data with icons and descriptions
        self.providers = {
            "Google (free unofficial)": {
                "display_name": "Google Translate",
                "description": "Free translation service from Google",
                "tags": ["Free", "Fast", "Basic"],
                "icon": "assets/providers/google.png",
                "category": "free",
                "short_name": "G"
            },
            "G4F: API (g4f.dev)": {
                "display_name": "G4F API",
                "description": "Free access to various AI models",
                "tags": ["Free", "Multiple Models", "Community"],
                "icon": "assets/providers/g4f.png",
                "category": "free",
                "short_name": "G4F"
            },
            "IO: chat.completions": {
                "display_name": "IO Intelligence",
                "description": "Affordable AI models with good quality",
                "tags": ["Affordable", "Fast", "Reliable"],
                "icon": "assets/providers/io.png",
                "category": "affordable",
                "short_name": "IO"
            },
            "OpenAI Compatible API": {
                "display_name": "OpenAI Compatible",
                "description": "Standard OpenAI API or compatible services",
                "tags": ["Premium", "GPT Models", "Industry Standard"],
                "icon": "assets/providers/openai.png",
                "category": "premium",
                "short_name": "OAI"
            },
            "Anthropic: Claude": {
                "display_name": "Anthropic Claude",
                "description": "Advanced AI with excellent reasoning",
                "tags": ["Premium", "Advanced", "Claude"],
                "icon": "assets/providers/anthropic.png",
                "category": "premium",
                "short_name": "A"
            },
            "Google: Gemini": {
                "display_name": "Google Gemini",
                "description": "Google's latest AI models",
                "tags": ["Premium", "Gemini", "Multimodal"],
                "icon": "assets/providers/gemini.png",
                "category": "premium",
                "short_name": "G"
            },
            "Yandex Translate": {
                "display_name": "Yandex Translate",
                "description": "Free translation from Yandex",
                "tags": ["Free", "Russian", "Basic"],
                "icon": "assets/providers/yandex.png",
                "category": "free",
                "short_name": "Я"
            },
            "Yandex Cloud": {
                "display_name": "Yandex Cloud AI",
                "description": "Advanced AI models from Yandex",
                "tags": ["Premium", "Russian", "Cloud"],
                "icon": "assets/providers/yandex.png",
                "category": "premium",
                "short_name": "Я"
            },
            "DeepL API": {
                "display_name": "DeepL API",
                "description": "Professional translation service",
                "tags": ["Premium", "Professional", "Accurate"],
                "icon": "assets/providers/deepl.png",
                "category": "premium",
                "short_name": "D"
            },
            "Fireworks.ai": {
                "display_name": "Fireworks.ai",
                "description": "Fast and affordable AI models",
                "tags": ["Affordable", "Fast", "Open Source"],
                "icon": "assets/providers/fireworks.png",
                "category": "affordable",
                "short_name": "F"
            },
            "Groq": {
                "display_name": "Groq",
                "description": "Ultra-fast AI inference",
                "tags": ["Ultra Fast", "Llama", "Efficient"],
                "icon": "assets/providers/groq.png",
                "category": "affordable",
                "short_name": "G"
            },
            "Together.ai": {
                "display_name": "Together.ai",
                "description": "Community-driven AI models",
                "tags": ["Community", "Open Source", "Affordable"],
                "icon": "assets/providers/together.png",
                "category": "affordable",
                "short_name": "T"
            },
            "Ollama": {
                "display_name": "Ollama",
                "description": "Local AI models for privacy",
                "tags": ["Local", "Private", "Open Source"],
                "icon": "assets/providers/ollama.png",
                "category": "local",
                "short_name": "O"
            },
            "Mistral AI": {
                "display_name": "Mistral AI",
                "description": "European AI with excellent performance",
                "tags": ["European", "Advanced", "Efficient"],
                "icon": "assets/providers/mistral.png",
                "category": "premium",
                "short_name": "M"
            }
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Set dialog background and styling with gradient
        self.setStyleSheet("""
            ProviderSelectorDialog {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 rgba(35, 35, 35, 0.98),
                    stop: 1 rgba(25, 25, 25, 0.98)
                );
                border: 2px solid #666666;
                border-radius: 15px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QWidget {
                background-color: transparent;
            }
            StrongBodyLabel {
                color: #ffffff;
            }
            BodyLabel {
                color: #cccccc;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = StrongBodyLabel("Choose Your AI Provider")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
        """)
        header_layout.addWidget(title)
        
        # Search with better styling
        self.search_edit = LineEdit()
        self.search_edit.setPlaceholderText("Search providers...")
        self.search_edit.textChanged.connect(self._filter_providers)
        self.search_edit.setStyleSheet("""
            LineEdit {
                background-color: rgba(50, 50, 50, 0.9);
                border: 2px solid #555555;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
            }
            LineEdit:focus {
                border-color: #777777;
                background-color: rgba(60, 60, 60, 0.95);
            }
        """)
        header_layout.addWidget(self.search_edit, 1)
        
        layout.addWidget(header)
        
        # Filter buttons with better styling
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setSpacing(10)
        
        self.filter_buttons = {}
        filters = ["All", "Free", "Affordable", "Premium", "Local"]
        
        for filter_name in filters:
            btn = PushButton(filter_name)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            if filter_name == "All":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, f=filter_name: self._filter_by_category(f))
            btn.setStyleSheet("""
                PushButton {
                    background-color: rgba(60, 60, 60, 0.8);
                    border: 1px solid #555555;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 500;
                    color: #cccccc;
                }
                PushButton:checked {
                    background-color: rgba(80, 120, 200, 0.9);
                    border-color: #4a90e2;
                    color: #ffffff;
                }
                PushButton:hover {
                    background-color: rgba(70, 70, 70, 0.9);
                    border-color: #666666;
                }
            """)
            filter_layout.addWidget(btn)
            self.filter_buttons[filter_name] = btn
        
        layout.addWidget(filter_widget)
        
        # Scroll area for providers with better styling
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #444444;
                border-radius: 10px;
                background-color: rgba(40, 40, 40, 0.9);
            }
            QScrollBar:vertical {
                background-color: rgba(50, 50, 50, 0.8);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }
        """)
        
        # Providers container
        self.providers_widget = QWidget()
        self.providers_widget.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        self.providers_layout = QVBoxLayout(self.providers_widget)
        self.providers_layout.setSpacing(12)
        self.providers_layout.setContentsMargins(15, 15, 15, 15)
        
        self._create_provider_cards()
        
        scroll.setWidget(self.providers_widget)
        layout.addWidget(scroll)
        
        # Bottom buttons with better styling
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 10, 0, 0)
        
        cancel_btn = PushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        cancel_btn.setStyleSheet("""
            PushButton {
                background-color: rgba(80, 80, 80, 0.9);
                border: 1px solid #666666;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 500;
                color: #ffffff;
                min-width: 80px;
            }
            PushButton:hover {
                background-color: rgba(100, 100, 100, 0.95);
                border-color: #888888;
            }
            PushButton:pressed {
                background-color: rgba(70, 70, 70, 0.9);
            }
        """)
        bottom_layout.addStretch()
        bottom_layout.addWidget(cancel_btn)
        
        layout.addWidget(bottom_widget)
    
    def _create_provider_cards(self):
        """Create provider cards."""
        # Clear existing cards
        while self.providers_layout.count():
            child = self.providers_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        for provider_key, provider_info in self.providers.items():
            card = ProviderCard(provider_key, provider_info)
            card.clicked.connect(self._on_provider_selected)
            self.providers_layout.addWidget(card)
        
        self.providers_layout.addStretch()
    
    def _filter_providers(self, text):
        """Filter providers by search text."""
        search_text = text.lower()
        
        for i in range(self.providers_layout.count()):
            widget = self.providers_layout.itemAt(i).widget()
            if isinstance(widget, ProviderCard):
                provider_info = self.providers[widget.provider_key]
                
                # Check if search text matches any part of provider info
                matches = (
                    search_text in provider_info['display_name'].lower() or
                    search_text in provider_info['description'].lower() or
                    any(search_text in tag.lower() for tag in provider_info.get('tags', []))
                )
                
                widget.setVisible(matches)
    
    def _filter_by_category(self, category):
        """Filter providers by category."""
        if category == "All":
            for i in range(self.providers_layout.count()):
                widget = self.providers_layout.itemAt(i).widget()
                if isinstance(widget, ProviderCard):
                    widget.setVisible(True)
        else:
            for i in range(self.providers_layout.count()):
                widget = self.providers_layout.itemAt(i).widget()
                if isinstance(widget, ProviderCard):
                    provider_info = self.providers[widget.provider_key]
                    widget.setVisible(provider_info.get('category') == category.lower())
    
    def _on_provider_selected(self, provider_key):
        """Handle provider selection."""
        self.provider_selected.emit(provider_key)
        self.close()