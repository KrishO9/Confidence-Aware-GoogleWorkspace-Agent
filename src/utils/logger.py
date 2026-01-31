"""
Logging configuration using loguru
"""

import sys
from loguru import logger
from pathlib import Path
from src.config import get_settings


def setup_logger():
    """Configure logger with file and console output"""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        settings.log_file,
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.log_level,
        backtrace=True,
        diagnose=True
    )
    
    return logger


def get_logger():
    """Get configured logger instance"""
    return logger

