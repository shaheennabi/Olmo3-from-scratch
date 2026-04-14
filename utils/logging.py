"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "olmo",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "olmo") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
