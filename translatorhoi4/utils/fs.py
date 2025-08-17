"""Filesystem helpers for localisation files."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from ..parsers.paradox_yaml import (
    LOCALISATION_LINE_RE,
    LANG_TAG_RE,
    SUPPORTED_LANG_HEADERS,
)
from ..translator.mask import TOKEN_FINDERS  # not used but imported for completeness


def collect_localisation_files(root: str) -> List[str]:
    exts = {'.yml', '.yaml'}
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def _combine_post_with_loc(post: str, add_marker: bool) -> str:
    post = post or ""
    if not add_marker:
        return post
    if "#LOC!" in post:
        return post
    if post.strip() == "":
        return " #LOC!"
    return f"{post} #LOC!"


def rename_filename_for_lang(filename: str, dst_lang: str) -> str:
    def _sub(m):
        return f"{m.group(1)}{dst_lang}"
    return LANG_TAG_RE.sub(_sub, filename)


def compute_output_path(src_path: str, cfg) -> str:
    if cfg.in_place:
        base_dir = os.path.dirname(src_path)
    else:
        rel = os.path.relpath(src_path, cfg.src_dir)
        base_dir = os.path.join(cfg.out_dir, os.path.dirname(rel))
    fname = os.path.basename(src_path)
    if cfg.rename_files:
        new_fname = rename_filename_for_lang(fname, cfg.dst_lang)
        if new_fname == fname:
            root, ext = os.path.splitext(fname)
            if ext.lower() in ('.yml', '.yaml'):
                new_fname = f"{root}_l_{cfg.dst_lang}{ext}"
        fname = new_fname
    return os.path.join(base_dir, fname)


def _find_prev_localized_file(prev_root: str, current_relpath: str, dst_lang: str) -> Optional[str]:
    rel_dir = os.path.dirname(current_relpath)
    cur_base = os.path.basename(current_relpath)
    candidate_base = rename_filename_for_lang(cur_base, dst_lang)
    candidate_path = os.path.join(prev_root, rel_dir, candidate_base)
    if os.path.isfile(candidate_path):
        return candidate_path
    try_dir = os.path.join(prev_root, rel_dir)
    if not os.path.isdir(try_dir):
        return None
    desired_root = re.sub(r'_l_[^\.]+(\.ya?ml)$', r'\1', cur_base, flags=re.IGNORECASE)
    for fn in os.listdir(try_dir):
        if not fn.lower().endswith(('.yml', '.yaml')):
            continue
        if re.sub(r'_l_[^\.]+(\.ya?ml)$', r'\1', fn, flags=re.IGNORECASE).lower() == desired_root.lower():
            return os.path.join(try_dir, fn)
    return None


def _build_prev_map(prev_file_path: str) -> Dict[str, Tuple[str, str]]:
    m: Dict[str, Tuple[str, str]] = {}
    try:
        with open(prev_file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            for line in f:
                mm = LOCALISATION_LINE_RE.match(line)
                if not mm:
                    continue
                _pre, key, _ver, text, post = mm.groups()
                if post and "#LOC!" in post:
                    m[key] = (text, post)
    except Exception:
        pass
    return m
