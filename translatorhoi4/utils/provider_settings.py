"""Provider settings helper functions.

These functions use provider_config.py to save/load provider settings
in a centralized and consistent manner.
"""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .provider_config import PROVIDER_CONFIGS

if TYPE_CHECKING:
    # Import UI class only for type checking to avoid circular imports
    from ..ui.ui_interfaces import MainWindow


def save_provider_settings(window: MainWindow, data: Dict[str, Any]) -> None:
    """Save all provider settings to data dictionary.
    
    Args:
        window: MainWindow instance
        data: Dictionary to save settings into
    """
    for provider_key, config in PROVIDER_CONFIGS.items():
        # Save regular settings
        for setting in config.settings:
            widget = getattr(window, setting.widget_attr, None)
            if widget is None:
                continue
                
            if setting.is_checked:
                # CheckBox/SwitchButton
                data[setting.key] = widget.isChecked()
            else:
                # LineEdit/ComboBox/SpinBox
                if hasattr(widget, 'currentText'):
                    data[setting.key] = widget.currentText()
                elif hasattr(widget, 'value'):
                    data[setting.key] = widget.value()
                else:
                    data[setting.key] = widget.text()
        
        # Save cost settings
        if config.cost_input_key and config.cost_input_widget:
            widget = getattr(window, config.cost_input_widget, None)
            if widget:
                data[config.cost_input_key] = widget.text()
        
        if config.cost_output_key and config.cost_output_widget:
            widget = getattr(window, config.cost_output_widget, None)
            if widget:
                data[config.cost_output_key] = widget.text()


def load_provider_settings(window: MainWindow, settings: Dict[str, Any]) -> None:
    """Load all provider settings from settings dictionary.
    
    Args:
        window: MainWindow instance
        settings: Dictionary to load settings from
    """
    for provider_key, config in PROVIDER_CONFIGS.items():
        # Load regular settings
        for setting in config.settings:
            widget = getattr(window, setting.widget_attr, None)
            if widget is None:
                continue
            
            value = settings.get(setting.key)
            if value is None:
                continue
                
            if setting.is_checked:
                # CheckBox/SwitchButton
                widget.setChecked(bool(value))
            else:
                # LineEdit/ComboBox/SpinBox
                if hasattr(widget, 'setCurrentText'):
                    if value:
                        widget.setCurrentText(str(value))
                elif hasattr(widget, 'setValue'):
                    if isinstance(value, (int, float)):
                        widget.setValue(int(value))
                else:
                    widget.setText(str(value))
        
        # Load cost settings
        if config.cost_input_key and config.cost_input_widget:
            widget = getattr(window, config.cost_input_widget, None)
            if widget:
                value = settings.get(config.cost_input_key, config.default_input_cost)
                widget.setText(str(value))
        
        if config.cost_output_key and config.cost_output_widget:
            widget = getattr(window, config.cost_output_widget, None)
            if widget:
                value = settings.get(config.cost_output_key, config.default_output_cost)
                widget.setText(str(value))
