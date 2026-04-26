"""Provider configuration definitions.

This module centralizes all provider-related configuration including:
- Provider metadata (name, display name, category)
- Setting keys for saving/loading
- Default values
- UI widget references

This makes it easy to add new providers and ensures consistency
across save/load operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProviderSetting:
    """Represents a single setting for a provider."""
    key: str  # JSON key for saving/loading
    widget_attr: str  # Widget attribute name in UI (e.g., 'ed_nvidia_api_key')
    default: Any = None  # Default value
    is_checked: bool = False  # True if widget uses isChecked(), False if text()


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    registry_key: str  # Key in MODEL_REGISTRY
    display_name: str  # Human-readable name
    category: str  # Category for UI grouping (e.g., 'cloud', 'local', 'free')
    
    # Settings specific to this provider
    settings: List[ProviderSetting] = field(default_factory=list)
    
    # Container widget attribute (for show/hide)
    container_attr: Optional[str] = None
    
    # Model selector widget (if provider has model selection)
    model_widget_attr: Optional[str] = None
    
    # Cost settings (input/output per million tokens)
    cost_input_key: Optional[str] = None  # JSON key for input cost
    cost_output_key: Optional[str] = None  # JSON key for output cost
    cost_input_widget: Optional[str] = None  # Widget for input cost
    cost_output_widget: Optional[str] = None  # Widget for output cost
    default_input_cost: str = "0.0"
    default_output_cost: str = "0.0"


# ============================================================================
# PROVIDER CONFIGURATIONS
# ============================================================================

PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    'g4f': ProviderConfig(
        registry_key='G4F: API (g4f.dev)',
        display_name='G4F API',
        category='cloud',
        container_attr='g4f_container',
        settings=[
            ProviderSetting('g4f_model', 'ed_g4f_model', 'gpt-4o'),
            ProviderSetting('g4f_api_key', 'ed_g4f_api_key'),
            ProviderSetting('g4f_async', 'chk_g4f_async', True, is_checked=True),
            ProviderSetting('g4f_cc', 'spn_g4f_cc', 12),
        ],
        cost_input_key='g4f_input_cost',
        cost_output_key='g4f_output_cost',
        cost_input_widget='g4f_input',
        cost_output_widget='g4f_output',
    ),
    
    'io': ProviderConfig(
        registry_key='IO: chat.completions',
        display_name='IO Intelligence',
        category='cloud',
        container_attr='io_container',
        settings=[
            ProviderSetting('io_api_key', 'ed_io_api_key'),
            ProviderSetting('io_base_url', 'ed_io_base', 'https://api.intelligence.io.solutions/api/v1/'),
            ProviderSetting('io_async', 'chk_io_async', True, is_checked=True),
            ProviderSetting('io_cc', 'spn_io_cc', 12),
        ],
        cost_input_key='io_input_cost',
        cost_output_key='io_output_cost',
        cost_input_widget='io_input',
        cost_output_widget='io_output',
        default_input_cost='0.59',
        default_output_cost='0.79',
    ),
    
    'openai': ProviderConfig(
        registry_key='OpenAI Compatible API',
        display_name='OpenAI Compatible',
        category='cloud',
        container_attr='openai_container',
        settings=[
            ProviderSetting('openai_api_key', 'ed_openai_api_key'),
            ProviderSetting('openai_base_url', 'ed_openai_base', 'https://api.openai.com/v1/'),
            ProviderSetting('openai_model', 'ed_openai_model', 'gpt-4'),
            ProviderSetting('openai_async', 'chk_openai_async', True, is_checked=True),
            ProviderSetting('openai_cc', 'spn_openai_cc', 12),
        ],
        cost_input_key='openai_input_cost',
        cost_output_key='openai_output_cost',
        cost_input_widget='openai_input',
        cost_output_widget='openai_output',
        default_input_cost='2.50',
        default_output_cost='10.00',
    ),
    
    'anthropic': ProviderConfig(
        registry_key='Anthropic: Claude',
        display_name='Anthropic (Claude)',
        category='cloud',
        container_attr='anthropic_container',
        settings=[
            ProviderSetting('anthropic_api_key', 'ed_anthropic_api_key'),
            ProviderSetting('anthropic_model', 'ed_anthropic_model', 'claude-sonnet-4-5-20250929'),
            ProviderSetting('anthropic_async', 'chk_anthropic_async', True, is_checked=True),
            ProviderSetting('anthropic_cc', 'spn_anthropic_cc', 12),
        ],
        cost_input_key='anthropic_input_cost',
        cost_output_key='anthropic_output_cost',
        cost_input_widget='anthropic_input',
        cost_output_widget='anthropic_output',
        default_input_cost='3.00',
        default_output_cost='15.00',
    ),
    
    'gemini': ProviderConfig(
        registry_key='Google: Gemini',
        display_name='Google Gemini',
        category='cloud',
        container_attr='gemini_container',
        settings=[
            ProviderSetting('gemini_api_key', 'ed_gemini_api_key'),
            ProviderSetting('gemini_model', 'ed_gemini_model', 'gemini-2.5-flash'),
            ProviderSetting('gemini_async', 'chk_gemini_async', True, is_checked=True),
            ProviderSetting('gemini_cc', 'spn_gemini_cc', 12),
        ],
        cost_input_key='gemini_input_cost',
        cost_output_key='gemini_output_cost',
        cost_input_widget='gemini_input',
        cost_output_widget='gemini_output',
        default_input_cost='0.125',
        default_output_cost='0.375',
    ),
    
    'yandex_translate': ProviderConfig(
        registry_key='Yandex Translate',
        display_name='Yandex Translate',
        category='cloud',
        container_attr='yandex_translate_container',
        settings=[
            ProviderSetting('yandex_translate_api_key', 'ed_yandex_translate_api_key'),
            ProviderSetting('yandex_iam_token', 'ed_yandex_iam_token'),
            ProviderSetting('yandex_folder_id', 'ed_yandex_folder_id', 'b1g20dtckjkooop0futg'),
            ProviderSetting('yandex_translate_async', 'chk_yandex_translate_async', True, is_checked=True),
            ProviderSetting('yandex_translate_cc', 'spn_yandex_translate_cc', 12),
        ],
        cost_input_key='yandex_translate_input_cost',
        cost_output_key='yandex_translate_output_cost',
        cost_input_widget='yandex_translate_input',
        cost_output_widget='yandex_translate_output',
    ),
    
    'yandex_cloud': ProviderConfig(
        registry_key='Yandex Cloud',
        display_name='Yandex Cloud',
        category='cloud',
        container_attr='yandex_cloud_container',
        settings=[
            ProviderSetting('yandex_cloud_api_key', 'ed_yandex_cloud_api_key'),
            ProviderSetting('yandex_cloud_model', 'ed_yandex_cloud_model', 'aliceai-llm/latest'),
            ProviderSetting('yandex_async', 'chk_yandex_async', True, is_checked=True),
            ProviderSetting('yandex_cc', 'spn_yandex_cc', 12),
        ],
        cost_input_key='yandex_cloud_input_cost',
        cost_output_key='yandex_cloud_output_cost',
        cost_input_widget='yandex_cloud_input',
        cost_output_widget='yandex_cloud_output',
    ),
    
    'deepl': ProviderConfig(
        registry_key='DeepL API',
        display_name='DeepL API',
        category='cloud',
        container_attr='deepl_container',
        settings=[
            ProviderSetting('deepl_api_key', 'ed_deepl_api_key'),
            ProviderSetting('deepl_async', 'chk_deepl_async', True, is_checked=True),
            ProviderSetting('deepl_cc', 'spn_deepl_cc', 12),
        ],
        cost_input_key='deepl_input_cost',
        cost_output_key='deepl_output_cost',
        cost_input_widget='deepl_input',
        cost_output_widget='deepl_output',
    ),
    
    'fireworks': ProviderConfig(
        registry_key='Fireworks.ai',
        display_name='Fireworks.ai',
        category='cloud',
        container_attr='fireworks_container',
        settings=[
            ProviderSetting('fireworks_api_key', 'ed_fireworks_api_key'),
            ProviderSetting('fireworks_model', 'ed_fireworks_model', 'accounts/fireworks/models/llama-v3p1-8b-instruct'),
            ProviderSetting('fireworks_async', 'chk_fireworks_async', True, is_checked=True),
            ProviderSetting('fireworks_cc', 'spn_fireworks_cc', 12),
        ],
        cost_input_key='fireworks_input_cost',
        cost_output_key='fireworks_output_cost',
        cost_input_widget='fireworks_input',
        cost_output_widget='fireworks_output',
    ),
    
    'groq': ProviderConfig(
        registry_key='Groq',
        display_name='Groq',
        category='cloud',
        container_attr='groq_container',
        settings=[
            ProviderSetting('groq_api_key', 'ed_groq_api_key'),
            ProviderSetting('groq_model', 'ed_groq_model', 'openai/gpt-oss-20b'),
            ProviderSetting('groq_async', 'chk_groq_async', True, is_checked=True),
            ProviderSetting('groq_cc', 'spn_groq_cc', 12),
        ],
        cost_input_key='groq_input_cost',
        cost_output_key='groq_output_cost',
        cost_input_widget='groq_input',
        cost_output_widget='groq_output',
    ),
    
    'together': ProviderConfig(
        registry_key='Together.ai',
        display_name='Together.ai',
        category='cloud',
        container_attr='together_container',
        settings=[
            ProviderSetting('together_api_key', 'ed_together_api_key'),
            ProviderSetting('together_model', 'ed_together_model', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'),
            ProviderSetting('together_async', 'chk_together_async', True, is_checked=True),
            ProviderSetting('together_cc', 'spn_together_cc', 12),
        ],
        cost_input_key='together_input_cost',
        cost_output_key='together_output_cost',
        cost_input_widget='together_input',
        cost_output_widget='together_output',
    ),
    
    'ollama': ProviderConfig(
        registry_key='Ollama',
        display_name='Ollama',
        category='local',
        container_attr='ollama_container',
        settings=[
            ProviderSetting('ollama_model', 'ed_ollama_model', 'llama3.2'),
            ProviderSetting('ollama_base_url', 'ed_ollama_base_url', 'http://localhost:11434'),
            ProviderSetting('ollama_async', 'chk_ollama_async', True, is_checked=True),
            ProviderSetting('ollama_cc', 'spn_ollama_cc', 12),
        ],
        cost_input_key='ollama_input_cost',
        cost_output_key='ollama_output_cost',
        cost_input_widget='ollama_input',
        cost_output_widget='ollama_output',
    ),
    
    'mistral': ProviderConfig(
        registry_key='Mistral AI',
        display_name='Mistral AI',
        category='cloud',
        container_attr='mistral_container',
        settings=[
            ProviderSetting('mistral_api_key', 'ed_mistral_api_key'),
            ProviderSetting('mistral_model', 'ed_mistral_model', 'mistral-small-latest'),
            ProviderSetting('mistral_async', 'chk_mistral_async', True, is_checked=True),
            ProviderSetting('mistral_cc', 'spn_mistral_cc', 12),
        ],
        cost_input_key='mistral_input_cost',
        cost_output_key='mistral_output_cost',
        cost_input_widget='mistral_input',
        cost_output_widget='mistral_output',
    ),
    
    'nvidia': ProviderConfig(
        registry_key='Nvidia NIM',
        display_name='Nvidia NIM',
        category='cloud',
        container_attr='nvidia_container',
        settings=[
            ProviderSetting('nvidia_api_key', 'ed_nvidia_api_key'),
            ProviderSetting('nvidia_model', 'ed_nvidia_model', 'moonshotai/kimi-k2.5'),
            ProviderSetting('nvidia_base_url', 'ed_nvidia_base_url', 'https://integrate.api.nvidia.com/v1/chat/completions'),
            ProviderSetting('nvidia_async', 'chk_nvidia_async', True, is_checked=True),
            ProviderSetting('nvidia_cc', 'spn_nvidia_cc', 12),
        ],
        cost_input_key='nvidia_input_cost',
        cost_output_key='nvidia_output_cost',
        cost_input_widget='nvidia_input',
        cost_output_widget='nvidia_output',
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_provider_by_registry_key(registry_key: str) -> Optional[ProviderConfig]:
    """Get provider config by MODEL_REGISTRY key."""
    for config in PROVIDER_CONFIGS.values():
        if config.registry_key == registry_key:
            return config
    return None


def get_all_provider_keys() -> List[str]:
    """Get all provider keys."""
    return list(PROVIDER_CONFIGS.keys())


def get_provider_display_names() -> List[str]:
    """Get all provider display names (registry keys)."""
    return [config.registry_key for config in PROVIDER_CONFIGS.values()]
