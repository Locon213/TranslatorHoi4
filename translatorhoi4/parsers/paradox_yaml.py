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
