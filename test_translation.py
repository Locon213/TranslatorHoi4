#!/usr/bin/env python3
"""Test script for translation output path fixes."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from translatorhoi4.utils.fs import compute_output_path, rename_filename_for_lang

class MockConfig:
    """Mock configuration object for testing."""
    def __init__(self, **kwargs):
        self.in_place = kwargs.get('in_place', False)
        self.src_dir = kwargs.get('src_dir', '')
        self.out_dir = kwargs.get('out_dir', '')
        self.dst_lang = kwargs.get('dst_lang', 'russian')
        self.rename_files = kwargs.get('rename_files', True)
        self.mod_name = kwargs.get('mod_name', None)
        self.use_mod_name = kwargs.get('use_mod_name', False)

def test_rename_filename_for_lang():
    """Test filename renaming functionality."""
    print("Testing rename_filename_for_lang function...")
    
    test_cases = [
        # Стандартные теги _l_
        ("test_file_l_english.yml", "russian", "test_file_l_russian.yml"),
        ("events_l_french.yml", "german", "events_l_german.yml"),
        
        # Теги i_english НЕ должны заменяться
        ("test_file_i_english.yml", "russian", "test_file_i_english.yml"),
        ("interface_i_french.yml", "german", "interface_i_french.yml"),
        
        # Просто название языка в имени файла
        ("english_events.yml", "russian", "russian_events.yml"),
        ("french_interface.yml", "german", "german_interface.yml"),
        
        # Файлы без языковых тегов
        ("test_file.yml", "russian", "test_file.yml"),
        ("events.yml", "german", "events.yml"),
    ]
    
    for input_name, dst_lang, expected in test_cases:
        result = rename_filename_for_lang(input_name, dst_lang)
        status = "OK" if result == expected else "FAIL"
        print(f"  {status} {input_name} -> {result} (expected: {expected})")
        if result != expected:
            print(f"    ERROR: Expected '{expected}', got '{result}'")

def test_compute_output_path():
    """Test output path computation."""
    print("\nTesting compute_output_path function...")
    
    # Создаем временные директории для тестов
    with tempfile.TemporaryDirectory() as temp_dir:
        # Структура тестовых файлов
        test_structure = {
            '345/localisation/english/test_file_l_english.yml': 'l_english:\n  test_key:0 "Test"',
            'mod_folder/localisation/english/events_l_english.yml': 'l_english:\n  event_key:0 "Event"',
            'simple/test.yml': 'l_english:\n  simple_key:0 "Simple"',
        }
        
        # Создаем тестовые файлы
        for file_path, content in test_structure.items():
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Тестовые случаи
        test_cases = [
            # Базовый случай - перевод в указанную выходную директорию
            {
                'src_path': os.path.join(temp_dir, '345/localisation/english/test_file_l_english.yml'),
                'config': MockConfig(
                    src_dir=os.path.join(temp_dir, '345'),
                    out_dir=os.path.join(temp_dir, 'output'),
                    dst_lang='russian'
                ),
                'expected_path': os.path.join(temp_dir, 'output', 'localisation', 'russian', 'test_file_l_russian.yml')
            },
            
            # С использованием имени мода
            {
                'src_path': os.path.join(temp_dir, '345/localisation/english/test_file_l_english.yml'),
                'config': MockConfig(
                    src_dir=os.path.join(temp_dir, '345'),
                    out_dir=os.path.join(temp_dir, 'output'),
                    dst_lang='russian',
                    use_mod_name=True,
                    mod_name='MyMod'
                ),
                'expected_path': os.path.join(temp_dir, 'output', 'MyMod', 'localisation', 'russian', 'test_file_l_russian.yml')
            },
            
            # Простая структура без вложенных папок
            {
                'src_path': os.path.join(temp_dir, 'simple/test.yml'),
                'config': MockConfig(
                    src_dir=os.path.join(temp_dir, 'simple'),
                    out_dir=os.path.join(temp_dir, 'output'),
                    dst_lang='german'
                ),
                'expected_path': os.path.join(temp_dir, 'output', 'localisation', 'german', 'test_l_german.yml')
            },
            
            # In-place режим
            {
                'src_path': os.path.join(temp_dir, '345/localisation/english/test_file_l_english.yml'),
                'config': MockConfig(
                    src_dir=os.path.join(temp_dir, '345'),
                    out_dir='',
                    dst_lang='russian',
                    in_place=True
                ),
                'expected_path': os.path.join(temp_dir, '345/localisation/english/test_file_l_russian.yml')
            },
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                result = compute_output_path(test_case['src_path'], test_case['config'])
                expected = test_case['expected_path']
                
                # Нормализуем пути для сравнения (игнорируем разницу между / и \)
                result_norm = os.path.normpath(result)
                expected_norm = os.path.normpath(expected)
                
                # Проверяем, что путь соответствует ожидаемому
                status = "OK" if result_norm == expected_norm else "FAIL"
                print(f"  Test {i+1}: {status}")
                print(f"    Source: {test_case['src_path']}")
                print(f"    Result: {result}")
                print(f"    Expected: {expected}")
                
                if result_norm != expected_norm:
                    print(f"    ERROR: Path mismatch!")
                    print(f"    Normalized result: {result_norm}")
                    print(f"    Normalized expected: {expected_norm}")
                
                # Проверяем, что нет дублирования папок
                if 'localisation' in result:
                    path_parts = result.split(os.sep)
                    localisation_count = path_parts.count('localisation')
                    if localisation_count > 1:
                        print(f"    WARNING: Found {localisation_count} 'localisation' folders (should be 1)")
                
            except Exception as e:
                print(f"  Test {i+1}: FAIL ERROR: {e}")

def main():
    """Run all tests."""
    print("=== Testing Translation Output Path Fixes ===\n")
    
    test_rename_filename_for_lang()
    test_compute_output_path()
    
    print("\n=== Tests completed ===")

if __name__ == "__main__":
    main()