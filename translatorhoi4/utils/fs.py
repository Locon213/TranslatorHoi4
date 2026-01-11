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
    
    # Заменяем только стандартный тег _l_
    result = LANG_TAG_RE.sub(_sub, filename)
    
    # Если замена не произошла, пробуем заменить просто название языка в имени файла
    if result == filename:
        # Заменяем english, russian и т.д. в имени файла, но только если это не часть другого тега
        # Сначала пробуем в конце имени файла (перед расширением)
        pattern = r'\b(english|russian|german|french|spanish|braz_por|polish|japanese|korean|simp_chinese)\b(?=\.(yml|yaml)$)'
        result = re.sub(pattern, dst_lang, filename, flags=re.IGNORECASE)
        
        # Если все еще не заменено, пробуем в начале имени файла
        if result == filename:
            pattern = r'^(english|russian|german|french|spanish|braz_por|polish|japanese|korean|simp_chinese)(?=_)(.+?)(\.yml|\.yaml)$'
            match = re.match(pattern, filename, flags=re.IGNORECASE)
            if match:
                lang, middle, ext = match.groups()
                result = f"{dst_lang}{middle}{ext}"
    
    return result


def compute_output_path(src_path: str, cfg) -> str:
    if cfg.in_place:
        base_dir = os.path.dirname(src_path)
    else:
        # Используем выходную директорию, указанную пользователем
        base_dir = cfg.out_dir
        
        # Если указано использовать имя мода, добавляем его
        if cfg.use_mod_name and cfg.mod_name:
            base_dir = os.path.join(base_dir, cfg.mod_name)
        
        # Добавляем языковую директорию
        base_dir = os.path.join(base_dir, "localisation", cfg.dst_lang)
        
        # Добавляем относительный путь из исходной директории, но без дублирования
        rel = os.path.relpath(src_path, cfg.src_dir)
        rel_dir = os.path.dirname(rel)
        if rel_dir and rel_dir != '.':
            # Проверяем, не содержит ли уже путь нужную структуру
            if 'localisation' not in rel_dir.lower():
                base_dir = os.path.join(base_dir, rel_dir)
    
    # Add logging to debug directory creation
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Creating directory: {base_dir}")
    
    fname = os.path.basename(src_path)
    if cfg.rename_files:
        new_fname = rename_filename_for_lang(fname, cfg.dst_lang)
        if new_fname == fname:
            root, ext = os.path.splitext(fname)
            if ext.lower() in ('.yml', '.yaml'):
                new_fname = f"{root}_l_{cfg.dst_lang}{ext}"
        fname = new_fname
    # Ensure the directory exists
    try:
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Successfully created directory: {base_dir}")
    except Exception as e:
        logger.error(f"Failed to create directory {base_dir}: {e}")
        raise
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
