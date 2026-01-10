"""Enhanced logging with file rotation and multiple log levels."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

# Try to import loguru for better logging, fall back to standard logging
try:
    from loguru import logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False


class LogManager:
    """Manage application logging with rotation and multiple outputs."""
    
    _instance: Optional["LogManager"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "LogManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._log_dir: Optional[Path] = None
        self._log_files: dict = {}
        self._standard_logger: Optional[logging.Logger] = None
    
    def setup(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "7 days",
        format_string: Optional[str] = None,
    ):
        """Setup logging with rotation.
        
        Args:
            log_dir: Directory for log files (default: logs/ in app directory)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            rotation: Size or time for log rotation (e.g., "10 MB", "1 hour", "midnight")
            retention: How long to keep old logs (e.g., "7 days", "1 week")
            format_string: Custom format string for log messages
        """
        if log_dir is None:
            # Default to logs directory in app folder
            app_dir = Path(__file__).parent.parent
            log_dir = app_dir / "logs"
        
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        if _HAS_LOGURU:
            self._setup_loguru(level, rotation, retention, format_string)
        else:
            self._setup_standard(level, rotation, retention, format_string)
    
    def _setup_loguru(
        self,
        level: str,
        rotation: str,
        retention: str,
        format_string: Optional[str],
    ):
        """Setup loguru-based logging."""
        # Remove default handler
        logger.remove()
        
        # Default format
        if format_string is None:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{message}</cyan>"
            )
        
        # Add main log file with rotation
        log_path = self._log_dir / "app.log"
        logger.add(
            str(log_path),
            level=level,
            rotation=rotation,
            retention=retention,
            format=format_string,
            encoding="utf-8",
        )
        
        # Add error-only log file
        error_log_path = self._log_dir / "error.log"
        logger.add(
            str(error_log_path),
            level="ERROR",
            rotation=rotation,
            retention=retention,
            format=format_string,
            encoding="utf-8",
        )
        
        # Add debug log for debugging purposes
        debug_log_path = self._log_dir / "debug.log"
        logger.add(
            str(debug_log_path),
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            format=format_string,
            encoding="utf-8",
        )
        
        # Add console output
        logger.add(
            lambda msg: print(msg, end=""),
            format=format_string.replace("<green>", "").replace("</green>", "")
            .replace("<level>", "").replace("</level>", "")
            .replace("<cyan>", "").replace("</cyan>", ""),
        )
    
    def _setup_standard(
        self,
        level: str,
        rotation: str,
        retention: str,
        format_string: Optional[str],
    ):
        """Setup standard logging module as fallback."""
        # Create standard logger
        self._standard_logger = logging.getLogger("translatorhoi4")
        self._standard_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Clear existing handlers
        self._standard_logger.handlers.clear()
        
        # Default format
        if format_string is None:
            format_string = "%(asctime)s | %(levelname)-8s | %(message)s"
        
        formatter = logging.Formatter(format_string)
        
        # File handler with basic rotation simulation
        log_path = self._log_dir / "app.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._standard_logger.addHandler(file_handler)
        
        # Error file handler
        error_log_path = self._log_dir / "error.log"
        error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self._standard_logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        self._standard_logger.addHandler(console_handler)
    
    def log(self, level: str, message: str, **kwargs):
        """Log a message with the given level."""
        if _HAS_LOGURU:
            log_level = getattr(logger, level.lower(), "INFO")
            logger.log(log_level, message, **kwargs)
        elif self._standard_logger:
            log_level = getattr(self._standard_logger, level.lower(), logging.INFO)
            self._standard_logger.log(log_level, message)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        if _HAS_LOGURU:
            logger.debug(message, **kwargs)
        elif self._standard_logger:
            self._standard_logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        if _HAS_LOGURU:
            logger.info(message, **kwargs)
        elif self._standard_logger:
            self._standard_logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        if _HAS_LOGURU:
            logger.warning(message, **kwargs)
        elif self._standard_logger:
            self._standard_logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        if _HAS_LOGURU:
            logger.error(message, **kwargs)
        elif self._standard_logger:
            self._standard_logger.error(message)
    
    def exception(self, message: str, **kwargs):
        """Log an exception with traceback."""
        if _HAS_LOGURU:
            logger.exception(message, **kwargs)
        elif self._standard_logger:
            self._standard_logger.exception(message)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        if _HAS_LOGURU:
            logger.critical(message, **kwargs)
        elif self._standard_logger:
            self._standard_logger.critical(message)
    
    @property
    def log_dir(self) -> Optional[Path]:
        """Get the log directory path."""
        return self._log_dir
    
    def get_log_files(self) -> dict:
        """Get list of log files with their paths and sizes."""
        if self._log_dir is None or not self._log_dir.exists():
            return {}
        
        files = {}
        for log_file in self._log_dir.glob("*.log"):
            try:
                size = log_file.stat().st_size
                modified = datetime.fromtimestamp(log_file.stat().st_mtime)
                files[log_file.name] = {
                    "path": str(log_file),
                    "size": size,
                    "size_human": self._human_size(size),
                    "modified": modified.isoformat(),
                }
            except Exception:
                pass
        return files
    
    def _human_size(self, size: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def clear_old_logs(self, older_than_days: int = 7):
        """Remove log files older than specified days."""
        if self._log_dir is None or not self._log_dir.exists():
            return
        
        cutoff = datetime.now() - timedelta(days=older_than_days)
        deleted = []
        
        for log_file in self._log_dir.glob("*.log"):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff:
                    log_file.unlink()
                    deleted.append(log_file.name)
            except Exception:
                pass
        
        return deleted
    
    def read_log(self, filename: str, max_lines: int = 1000) -> str:
        """Read contents of a log file."""
        if self._log_dir is None:
            return ""
        
        log_path = self._log_dir / filename
        if not log_path.exists():
            return ""
        
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    lines = lines[-max_lines:]
                return "".join(lines)
        except Exception:
            return ""


# Global log manager instance
log_manager = LogManager()


# Convenience function for simple logging
def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """Setup application logging with sensible defaults."""
    log_manager.setup(log_dir=log_dir, level=level, rotation=rotation, retention=retention)
    return log_manager
