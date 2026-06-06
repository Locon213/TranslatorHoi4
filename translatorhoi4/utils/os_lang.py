"""OS language detection utility."""
from __future__ import annotations

import os
import sys
import locale


def detect_app_language() -> str:
    """Detect primary language of the OS and map to program language code."""
    os_lang = "en"
    
    if sys.platform == "win32":
        try:
            import ctypes
            # GetUserDefaultUILanguage returns the language identifier (LCID)
            # e.g., 1049 (0x0419) for Russian
            lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            lcid_map = {
                1049: "ru",  # Russian
                1033: "en",  # English (US)
                2057: "en",  # English (UK)
                1031: "de",  # German
                1036: "fr",  # French
                3082: "es",  # Spanish (Modern Sort)
                1034: "es",  # Spanish (Traditional Sort)
                1046: "pt",  # Portuguese (Brazil)
                2070: "pt",  # Portuguese (Portugal)
                1045: "pl",  # Polish
                1041: "ja",  # Japanese
                1042: "ko",  # Korean
                2052: "zh",  # Chinese (PRC)
                1028: "zh",  # Chinese (Taiwan)
                1040: "it",  # Italian
                1066: "vi",  # Vietnamese
                1044: "no",  # Norwegian (Bokmål)
                2068: "no",  # Norwegian (Nynorsk)
            }
            if lcid in lcid_map:
                os_lang = lcid_map[lcid]
            else:
                # Primary language ID is the lower 10 bits of LCID
                primary_id = lcid & 0x3FF
                primary_map = {
                    0x19: "ru",
                    0x09: "en",
                    0x07: "de",
                    0x0C: "fr",
                    0x0A: "es",
                    0x16: "pt",
                    0x15: "pl",
                    0x11: "ja",
                    0x12: "ko",
                    0x04: "zh",
                    0x10: "it",
                    0x2A: "vi",
                    0x14: "no",
                }
                if primary_id in primary_map:
                    os_lang = primary_map[primary_id]
        except Exception:
            pass
    else:
        # macOS, Linux and fallbacks
        for env in ("LANG", "LC_ALL", "LC_CTYPE", "LANGUAGE"):
            val = os.environ.get(env)
            if val:
                lang = val.split("_")[0].split(".")[0].lower()
                if lang in ("ru", "en", "de", "fr", "es", "pt", "pl", "ja", "ko", "zh", "it", "vi", "no"):
                    os_lang = lang
                    break

        if os_lang == "en":
            try:
                # Standard Python locale check
                lang, _ = locale.getlocale()
                if lang:
                    lang = lang.split("_")[0].lower()
                    if lang in ("ru", "en", "de", "fr", "es", "pt", "pl", "ja", "ko", "zh", "it", "vi", "no"):
                        os_lang = lang
            except Exception:
                pass

    # Map the detected 2-letter OS language to program's internal language code
    mapping = {
        "ru": "russian",
        "en": "english",
        "de": "german",
        "fr": "french",
        "es": "spanish",
        "pt": "braz_por",
        "pl": "polish",
        "ja": "japanese",
        "ko": "korean",
        "zh": "simp_chinese",
        "it": "italian",
        "vi": "vietnamese",
        "no": "norwegian",
    }
    return mapping.get(os_lang, "english")
