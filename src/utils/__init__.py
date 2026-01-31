"""Utility functions and helpers"""

from .logger import setup_logger, get_logger
from .text_processing import extract_text_from_html, clean_email_text, chunk_text

__all__ = [
    "setup_logger",
    "get_logger", 
    "extract_text_from_html",
    "clean_email_text",
    "chunk_text"
]

