"""
Text processing utilities for email content
"""

import re
from bs4 import BeautifulSoup
import html2text
from typing import List


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML email content
    
    Args:
        html_content: HTML string
        
    Returns:
        Clean text string
    """
    if not html_content:
        return ""
    
    # Use html2text for better formatting
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    
    try:
        text = h.handle(html_content)
        return text.strip()
    except Exception:
        # Fallback to BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)


def clean_email_text(text: str) -> str:
    """
    Clean and normalize email text
    
    Args:
        text: Raw email text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove email signatures (common patterns)
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Sent from my .*', '', text, flags=re.IGNORECASE)
    
    # Remove quoted replies
    text = re.sub(r'On .* wrote:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'>.*', '', text, flags=re.MULTILINE)
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for embedding
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            next_period = text.find('.', end - 100, end + 100)
            if next_period != -1:
                end = next_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def extract_email_metadata(email_text: str) -> dict:
    """
    Extract metadata from email text (dates, names, etc.)
    
    Args:
        email_text: Email content
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {
        'urls': [],
        'emails': [],
        'phone_numbers': [],
        'dates': []
    }
    
    # Extract URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    metadata['urls'] = re.findall(url_pattern, email_text)
    
    # Extract email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    metadata['emails'] = re.findall(email_pattern, email_text)
    
    # Extract phone numbers (basic pattern)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    metadata['phone_numbers'] = re.findall(phone_pattern, email_text)
    
    # Extract dates (basic patterns)
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    metadata['dates'] = re.findall(date_pattern, email_text)
    
    return metadata

