"""Filesystem helpers for localisation files."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..parsers.paradox_yaml import (
    LOCALISATION_LINE_RE,
    LANG_TAG_RE,
    SUPPORTED_LANG_HEADERS,
)
from ..translator.mask import TOKEN_FINDERS  # not used but imported for completeness

# Pre-compiled regex for better performance
_LANG_NAME_RE = re.compile(r'\b(english|russian|german|french|spanish|braz_por|polish|japanese|korean|simp_chinese)\b', re.IGNORECASE)
_LANG_FOLDER_RE = re.compile(r'_l_[a-z]+', re.IGNORECASE)


@lru_cache(maxsize=128)
def _is_localisation_file(filename: str, extensions: frozenset) -> bool:
    """Check if file is a localisation file (cached for performance)."""
    return os.path.splitext(filename)[1].lower() in extensions


def collect_localisation_files(root: str) -> List[str]:
    """Collect all localisation files from root directory."""
    exts = frozenset(['.yml', '.yaml'])
    files: List[str] = []
    
    # Use os.scandir for better performance than os.walk
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if _is_localisation_file(fn, exts):
                files.append(os.path.join(dirpath, fn))
    
    files.sort()
    return files


def collect_localisation_files_parallel(root: str, max_workers: int = 4) -> List[str]:
    """Collect localisation files using parallel processing for large directories."""
    exts = frozenset(['.yml', '.yaml'])
    files: List[str] = []
    
    def scan_directory(dir_path: str) -> List[str]:
        """Scan a single directory for localisation files."""
        local_files = []
        try:
            for entry in os.scandir(dir_path):
                if entry.is_file() and _is_localisation_file(entry.name, exts):
                    local_files.append(entry.path)
                elif entry.is_dir():
                    # Recursively scan subdirectories
                    local_files.extend(scan_directory(entry.path))
        except (OSError, PermissionError):
            # Skip directories we can't access
            pass
        return local_files
    
    # For small directories, use regular walk
    try:
        dir_count = sum(1 for _ in os.walk(root))
        if dir_count < 50:  # Threshold for using parallel processing
            return collect_localisation_files(root)
    except:
        pass
    
    # Use ThreadPoolExecutor for parallel processing
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for top-level directories
            futures = []
            for entry in os.scandir(root):
                if entry.is_dir():
                    futures.append(executor.submit(scan_directory, entry.path))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    files.extend(future.result())
                except Exception:
                    pass
    except:
        # Fallback to regular method
        return collect_localisation_files(root)
    
    files.sort()
    return files


def collect_localisation_files_by_lang(root: str, target_languages: List[str],
                                     include_replace: bool = True) -> List[str]:
    """
    Collect localisation files only for specified target languages.
    
    Args:
        root: Root directory to search
        target_languages: List of target language codes (e.g., ['english', 'russian'])
        include_replace: Whether to include files from 'replace' folders
        
    Returns:
        List of file paths that match the criteria
    """
    exts = {'.yml', '.yaml'}
    files: List[str] = []
    target_langs_lower = [lang.lower() for lang in target_languages]
    
    for dirpath, _, filenames in os.walk(root):
        # Check if this is a replace folder
        is_replace_folder = 'replace' in dirpath.lower()
        
        # Skip replace folders if not requested
        if is_replace_folder and not include_replace:
            continue
            
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                file_path = os.path.join(dirpath, fn)
                
                # Check if file matches target languages
                file_lower = fn.lower()
                
                # Check for language tags in filename
                matches_target = False
                for lang in target_langs_lower:
                    # Check for _l_<lang> pattern
                    if f'_l_{lang}' in file_lower:
                        matches_target = True
                        break
                    # Check for language name in filename
                    elif lang in file_lower:
                        matches_target = True
                        break
                
                # Also include files that don't have language tags (base files)
                if not matches_target:
                    # Check if file has any language tag
                    has_any_lang_tag = bool(re.search(r'_l_[a-z]+', file_lower))
                    if not has_any_lang_tag:
                        # This might be a base file, include it
                        matches_target = True
                
                if matches_target:
                    files.append(file_path)
    
    files.sort()
    return files


def collect_source_language_files(root: str, source_lang: str) -> List[str]:
    """
    Collect only source language files for translation.
    
    Args:
        root: Root directory to search
        source_lang: Source language code (e.g., 'english')
        
    Returns:
        List of source language file paths
    """
    exts = {'.yml', '.yaml'}
    files: List[str] = []
    source_lang_lower = source_lang.lower()
    
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                file_path = os.path.join(dirpath, fn)
                file_lower = fn.lower()
                
                # Check if this is a source language file
                if f'_l_{source_lang_lower}' in file_lower or source_lang_lower in file_lower:
                    files.append(file_path)
    
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


@lru_cache(maxsize=1024)
def _analyze_path_structure(rel_dir: str, dst_lang: str) -> Tuple[bool, List[str], bool]:
    """Analyze path structure for localisation folders (cached for performance)."""
    path_parts = rel_dir.lower().split(os.sep) if rel_dir else []
    has_localisation = any('localisation' in part for part in path_parts)
    
    # Check if path already contains language folder
    lang_replaced = False
    new_parts = []
    
    for part in path_parts:
        if not lang_replaced and _LANG_NAME_RE.search(part):
            new_parts.append(dst_lang)
            lang_replaced = True
        else:
            new_parts.append(part)
    
    # Check if relative dir contains language names
    has_lang_in_rel = bool(_LANG_NAME_RE.search(rel_dir))
    
    return has_localisation, new_parts, has_lang_in_rel


def compute_output_path(src_path: str, cfg) -> str:
    """Compute output path for translated file, avoiding duplicate language folders."""
    if cfg.in_place:
        return src_path
    
    # Use output directory specified by user
    base_dir = cfg.out_dir
    
    # If mod name is specified, add it
    if cfg.use_mod_name and cfg.mod_name:
        base_dir = os.path.join(base_dir, cfg.mod_name)
    
    # Get relative path from source directory
    rel = os.path.relpath(src_path, cfg.src_dir)
    rel_dir = os.path.dirname(rel)
    
    # Use cached analysis for better performance
    has_localisation, new_parts, has_lang_in_rel = _analyze_path_structure(rel_dir, cfg.dst_lang)
    
    if has_localisation:
        # Path already has localisation structure, use it as-is but replace language
        if new_parts:
            base_dir = os.path.join(base_dir, *new_parts)
        else:
            base_dir = os.path.join(base_dir, "localisation", cfg.dst_lang)
    else:
        # No localisation structure in path, add it
        base_dir = os.path.join(base_dir, "localisation", cfg.dst_lang)
        
        # Add relative directory if it exists and doesn't contain language info
        if rel_dir and rel_dir != '.' and not has_lang_in_rel:
            base_dir = os.path.join(base_dir, rel_dir)
    
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
    except Exception as e:
        print(f"Warning: Failed to create directory {base_dir}: {e}")
    
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
