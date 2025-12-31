# app/utils/logging_config.py
"""
Logging configuration for Mission7 Credit Scoring API.
"""
import logging
import sys
from typing import Optional


_loggers = {}


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup and return a logger with consistent formatting.
    
    Args:
        name: Logger name (e.g., 'database', 'prediction', 'audit')
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(f"mission7.{name}")
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    _loggers[name] = logger
    return logger
