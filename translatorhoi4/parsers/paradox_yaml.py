"""Paradox localisation YAML helpers."""
from __future__ import annotations

import re

LOCALISATION_LINE_RE = re.compile(r'^(\s*)([A-Za-z0-9_.\-]+):\s*(\d+)?\s*"(.*)"(\s*(?:#.*)?)$')
HEADER_RE = re.compile(r'^\s*l_([a-z_]+)\s*:\s*$')
LANG_TAG_RE = re.compile(
    r'(_l_)(english|russian|german|french|spanish|braz_por|polish|japanese|korean|simp_chinese)(?=\.(yml|yaml)$)',
    re.IGNORECASE
)
SUPPORTED_LANG_HEADERS = {
    'english': 'l_english:',
    'russian': 'l_russian:',
    'german': 'l_german:',
    'french': 'l_french:',
    'spanish': 'l_spanish:',
    'braz_por': 'l_braz_por:',
    'polish': 'l_polish:',
    'japanese': 'l_japanese:',
    'korean': 'l_korean:',
    'simp_chinese': 'l_simp_chinese:'
}

LANG_NAME_LIST = [
    'english', 'russian', 'german', 'french', 'spanish',
    'braz_por', 'polish', 'japanese', 'korean', 'simp_chinese'
]

# Native names for UI language selector with flags
LANG_NATIVE_NAMES = {
    'english': '🇺🇸 English',
    'russian': '🇷🇺 Русский',
    'german': '🇩🇪 Deutsch',
    'french': '🇫🇷 Français',
    'spanish': '🇪🇸 Español',
    'braz_por': '🇧🇷 Português (Brasil)',
    'polish': '🇵🇱 Polski',
    'japanese': '🇯🇵 日本語',
    'korean': '🇰🇷 한국어',
    'simp_chinese': '🇨🇳 中文',
}

def get_native_language_name(code: str) -> str:
    """Get native name for a language code."""
    return LANG_NATIVE_NAMES.get(code.lower(), code)

def parse_yaml_file(file_path: str) -> List[Dict[str, str]]:
    """Parse a Paradox YAML localisation file into a list of dictionaries."""
    import os
    
    data = []
    
    if not os.path.exists(file_path):
        return data
    
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Match localisation line pattern
            m = LOCALISATION_LINE_RE.match(line)
            if m:
                pre, key, version, text, post = m.groups()
                data.append({
                    'key': key,
                    'original': text,
                    'translation': text  # Initially set to original
                })
                
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return data


def parse_source_and_translation(src_file: str, trans_file: str) -> List[Dict[str, str]]:
    """
    Parse source and translated files together to get proper original/translation pairs.
    Returns list with 'key', 'original' from source and 'translation' from translated file.
    """
    import os
    
    data = []
    
    # Parse source file
    src_map = {}
    if os.path.exists(src_file):
        try:
            with open(src_file, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = LOCALISATION_LINE_RE.match(line)
                if m:
                    pre, key, version, text, post = m.groups()
                    src_map[key] = text
        except Exception as e:
            print(f"Error parsing source file {src_file}: {e}")
    
    # Parse translated file
    trans_map = {}
    if os.path.exists(trans_file):
        try:
            with open(trans_file, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = LOCALISATION_LINE_RE.match(line)
                if m:
                    pre, key, version, text, post = m.groups()
                    trans_map[key] = text
        except Exception as e:
            print(f"Error parsing translated file {trans_file}: {e}")
    
    # Combine into result - prefer translation, fall back to original
    all_keys = set(src_map.keys()) | set(trans_map.keys())
    for key in sorted(all_keys):
        original = src_map.get(key, trans_map.get(key, ''))
        translation = trans_map.get(key, original)
        data.append({
            'key': key,
            'original': original,
            'translation': translation
        })
    
    return data
