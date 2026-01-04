# translatorhoi4/app.py
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

# Мы не импортируем MainWindow здесь, чтобы избежать инициализации графики
# до настройки платформы (QT_QPA_PLATFORM).

def _install_excepthook() -> None:
    # Определяем базовый путь для логов
    base = Path(sys.argv[0]).resolve().parent
    
    def handle(exc_type, exc, tb):
        # 1. Выводим ошибку в консоль (для CI/CD логов)
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        
        # 2. Пытаемся записать в файл (для локальной отладки)
        try:
            log = base / "error.log"
            with log.open("w", encoding="utf-8") as fh:
                traceback.print_exception(exc_type, exc, tb, file=fh)
        except Exception:
            pass
            
    sys.excepthook = handle

def _resource_path(rel: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel
    return Path(__file__).resolve().parent.parent / rel  

def main(argv: list[str] | None = None) -> None:
    _install_excepthook()

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run offscreen and exit")
    args = parser.parse_args(argv)

    # Настройка для CI/CD и Smoke тестов
    if args.smoke or os.environ.get("CI"):
        print("Mode: CI/Smoke - Using offscreen platform")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        # Отключаем масштабирование, чтобы избежать проблем с шрифтами в headless
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

    print("Initializing QApplication...")
    # Создаем QApplication ДО импорта тяжелых виджетов
    app = QApplication(sys.argv[:1] + (argv or []))
    
    # Установка иконки
    icon_path = _resource_path("assets/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    else:
        print(f"Warning: Icon not found at {icon_path}")

    # Импортируем MainWindow только сейчас
    print("Importing MainWindow...")
    try:
        from translatorhoi4.ui.main_window import MainWindow
    except ImportError as e:
        print("CRITICAL: Failed to import MainWindow.")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL: Error during import phase: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Creating MainWindow instance...")
    try:
        w = MainWindow()
    except Exception as e:
        print("CRITICAL: Failed to initialize MainWindow.")
        # Это покажет, если проблема в отсутствующих ресурсах (например, qss/theme)
        traceback.print_exc()
        sys.exit(1)

    if args.smoke:
        print("Smoke test: Moving window offscreen...")
        w.move(-10000, -10000)
        
        print("Smoke test: Showing window...")
        w.show()
        
        print("Smoke test: Starting event loop for 500ms...")
        # Увеличил таймер до 500мс для надежности инициализации
        QTimer.singleShot(500, lambda: (print("Smoke test: SUCCESS. Quitting..."), app.quit()))
        
        ret_code = app.exec()
        print(f"Smoke test finished with code: {ret_code}")
        sys.exit(ret_code)

    # Обычный запуск
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()