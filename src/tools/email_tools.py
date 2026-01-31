"""
Email Tools
LangChain-compatible tools for email operations
"""

from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from src.api import GmailClient
from src.utils import get_logger

logger = get_logger()


class EmailSearchInput(BaseModel):
    """Input for email search"""
    query: str = Field(description="Search query or Gmail query syntax")
    max_results: int = Field(default=10, description="Maximum number of results")
    sender: Optional[str] = Field(default=None, description="Filter by sender email")


class EmailTools:
    """Email operation tools for LangGraph agents"""
    
    def __init__(self):
        self.gmail_client = GmailClient()
    
    def search_emails(
        self,
        query: str = "",
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search emails with filters
        
        Args:
            query: Search query
            sender: Sender filter
            subject: Subject filter
            max_results: Maximum results
            
        Returns:
            List of matching emails
        """
        try:
            results = self.gmail_client.search_emails(
                query=query,
                sender=sender,
                subject=subject,
                max_results=max_results
            )
            
            logger.info(f"Found {len(results)} emails")
            return results
            
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            return []
    
    def get_recent_emails(self, days: int = 7, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent emails
        
        Args:
            days: Number of days to look back
            max_results: Maximum results
            
        Returns:
            List of recent emails
        """
        try:
            results = self.gmail_client.get_recent_emails(
                days=days,
                max_results=max_results
            )
            
            logger.info(f"Retrieved {len(results)} recent emails")
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent emails: {e}")
            return []
    
    def get_email_by_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific email by ID
        
        Args:
            email_id: Gmail message ID
            
        Returns:
            Email details
        """
        try:
            message = self.gmail_client.get_message(email_id)
            if message:
                return self.gmail_client.parse_message(message)
            return None
            
        except Exception as e:
            logger.error(f"Error getting email {email_id}: {e}")
            return None
    
    def list_emails_from_sender(
        self,
        sender: str,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List all emails from specific sender
        
        Args:
            sender: Sender email or name
            max_results: Maximum results
            
        Returns:
            List of emails
        """
        return self.search_emails(sender=sender, max_results=max_results)
    
    def find_emails_with_attachments(
        self,
        query: Optional[str] = None,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find emails with attachments
        
        Args:
            query: Additional search query
            max_results: Maximum results
            
        Returns:
            List of emails with attachments
        """
        try:
            results = self.gmail_client.search_emails(
                query=query or "",
                has_attachment=True,
                max_results=max_results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding emails with attachments: {e}")
            return []
    
    def summarize_email(self, email: Dict[str, Any]) -> str:
        """
        Create a summary of an email
        
        Args:
            email: Parsed email dict
            
        Returns:
            Email summary string
        """
        subject = email.get('subject', 'No Subject')
        sender = email.get('from', 'Unknown')
        date = email.get('date', 'Unknown Date')
        body = email.get('body_text', '')[:300]
        
        summary = f"""Subject: {subject}
From: {sender}
Date: {date}
Preview: {body}..."""
        
        return summary
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """
        Get tool descriptions for agent
        
        Returns:
            List of tool descriptions
        """
        return [
            {
                "name": "search_emails",
                "description": "Search emails with various filters including query, sender, and subject",
                "parameters": ["query", "sender", "subject", "max_results"]
            },
            {
                "name": "get_recent_emails",
                "description": "Get recent emails from the last N days",
                "parameters": ["days", "max_results"]
            },
            {
                "name": "get_email_by_id",
                "description": "Get specific email details by Gmail message ID",
                "parameters": ["email_id"]
            },
            {
                "name": "list_emails_from_sender",
                "description": "List all emails from a specific sender",
                "parameters": ["sender", "max_results"]
            },
            {
                "name": "find_emails_with_attachments",
                "description": "Find emails that have attachments",
                "parameters": ["query", "max_results"]
            }
        ]

