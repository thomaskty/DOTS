"""
Logger Module
Provides centralized logging functionality for the application.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """
    Custom logger class for application-wide logging.

    Provides console and file logging with configurable log levels.
    """

    def __init__(self, log_name: str, log_level: str = "INFO"):
        """
        Initialize logger with specified name and level.

        Args:
            log_name: Name identifier for the logger
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_name = log_name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(self.log_level)

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)

        # Formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add console handler
        self.logger.addHandler(console_handler)

        # File handler placeholder
        self.file_handler: Optional[logging.FileHandler] = None

    def file_handler_flush(self, base_path: str, log_filename: Optional[str] = None) -> None:
        """
        Add file handler to logger and flush logs to file.

        Args:
            base_path: Base directory path for log files
            log_filename: Optional custom log filename
        """
        # Create logs directory
        log_dir = Path(base_path) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{self.log_name}_{timestamp}.log"

        log_filepath = log_dir / log_filename

        # Remove existing file handler if any
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()

        # Create new file handler
        self.file_handler = logging.FileHandler(log_filepath, mode='a')
        self.file_handler.setLevel(self.log_level)

        # Formatter for file handler
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)

        # Add file handler
        self.logger.addHandler(self.file_handler)

        self.logger.info(f"Log file created: {log_filepath}")