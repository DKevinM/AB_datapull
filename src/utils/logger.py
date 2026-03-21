"""Centralized logging configuration"""
import logging
import sys
from config.settings import LOG_LEVEL

def setup_logger(name: str) -> logging.Logger:
    """Create a configured logger for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
