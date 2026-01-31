"""
Email Full Content Storage
Stores full email bodies separately from vector store for complete retrieval
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils import get_logger

logger = get_logger()


class EmailStorage:
    """
    Stores full email content separately from vector embeddings
    This ensures we can retrieve complete email bodies even when emails are chunked
    """
    
    def __init__(self, storage_path: str = "data/email_storage.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = self._load_storage()
    
    def _load_storage(self) -> Dict[str, Any]:
        """Load email storage from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading email storage: {e}")
            return {}
    
    def _save_storage(self):
        """Save email storage to disk"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._storage, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving email storage: {e}")
    
    def store_email(self, email_id: str, email_data: Dict[str, Any]):
        """
        Store full email content
        
        Args:
            email_id: Gmail message ID
            email_data: Complete email data including full body
        """
        try:
            self._storage[email_id] = {
                'email_id': email_id,
                'subject': email_data.get('subject', ''),
                'sender': email_data.get('from', ''),
                'recipients': email_data.get('to', ''),
                'cc': email_data.get('cc', ''),
                'date': email_data.get('date', ''),
                'body_text': email_data.get('body_text', ''),
                'body_html': email_data.get('body_html', ''),
                'attachments': email_data.get('attachments', []),
                'labels': email_data.get('labels', []),
                'thread_id': email_data.get('thread_id', ''),
                'summary': email_data.get('summary', ''),
                'category': email_data.get('category', ''),
                'stored_at': email_data.get('stored_at', '')
            }
            self._save_storage()
            logger.debug(f"Stored full email content: {email_id}")
        except Exception as e:
            logger.error(f"Error storing email {email_id}: {e}")
    
    def get_email(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full email content
        
        Args:
            email_id: Gmail message ID
            
        Returns:
            Complete email data or None
        """
        return self._storage.get(email_id)
    
    def has_email(self, email_id: str) -> bool:
        """Check if email is stored"""
        return email_id in self._storage
    
    def delete_email(self, email_id: str):
        """Delete email from storage"""
        if email_id in self._storage:
            del self._storage[email_id]
            self._save_storage()
            logger.debug(f"Deleted email from storage: {email_id}")
    
    def clear_all(self):
        """Clear all stored emails"""
        self._storage = {}
        self._save_storage()
        logger.info("Cleared all email storage")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            'total_emails': len(self._storage),
            'storage_path': str(self.storage_path),
            'storage_size_mb': self.storage_path.stat().st_size / (1024 * 1024) if self.storage_path.exists() else 0
        }

