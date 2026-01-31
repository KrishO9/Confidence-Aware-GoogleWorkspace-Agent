"""
Gmail API Client
Handles all Gmail operations including reading, searching, and managing emails
"""

import base64
import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.config import get_settings
from src.utils import get_logger, extract_text_from_html, clean_email_text
import dateutil.parser

logger = get_logger()


class GmailClient:
    """Client for Gmail API operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Gmail API using OAuth2"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.settings.google_token_path):
            try:
                creds = Credentials.from_authorized_user_file(
                    self.settings.google_token_path,
                    self.settings.google_scopes
                )
            except Exception as e:
                logger.warning(f"Error loading token: {e}")
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Error refreshing token: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.settings.google_credentials_path):
                    logger.error(f"Credentials file not found: {self.settings.google_credentials_path}")
                    logger.info("Please provide Google OAuth2 credentials file")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.settings.google_credentials_path,
                    self.settings.google_scopes
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open(self.settings.google_token_path, 'w') as token:
                token.write(creds.to_json())
        
        # Build service
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API client authenticated successfully")
    
    def list_messages(
        self,
        query: str = "",
        max_results: int = 100,
        label_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List messages matching query
        
        Args:
            query: Gmail search query (e.g., "from:john@example.com")
            max_results: Maximum number of messages to return
            label_ids: List of label IDs to filter by
            
        Returns:
            List of message dictionaries
        """
        try:
            if not self.service:
                logger.error("Gmail service not initialized")
                return []
            
            messages = []
            page_token = None
            
            while len(messages) < max_results:
                request_params = {
                    'userId': 'me',
                    'q': query,
                    'maxResults': min(500, max_results - len(messages)),
                    'pageToken': page_token
                }
                
                if label_ids:
                    request_params['labelIds'] = label_ids
                
                results = self.service.users().messages().list(**request_params).execute()
                
                batch_messages = results.get('messages', [])
                messages.extend(batch_messages)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            logger.info(f"Found {len(messages)} messages matching query: {query}")
            return messages[:max_results]
            
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            return []
    
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full message details
        
        Args:
            message_id: Gmail message ID
            
        Returns:
            Complete message dictionary
        """
        try:
            if not self.service:
                return None
            
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            return message
            
        except HttpError as e:
            logger.error(f"Error getting message {message_id}: {e}")
            return None
    
    def parse_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Gmail message into structured format
        
        Args:
            message: Raw Gmail message
            
        Returns:
            Parsed message with extracted fields
        """
        parsed = {
            'id': message['id'],
            'thread_id': message['threadId'],
            'labels': message.get('labelIds', []),
            'snippet': message.get('snippet', ''),
            'internal_date': message.get('internalDate', ''),
            'subject': '',
            'from': '',
            'to': '',
            'cc': '',
            'date': '',
            'body_text': '',
            'body_html': '',
            'attachments': []
        }
        
        # Extract headers
        headers = message.get('payload', {}).get('headers', [])
        for header in headers:
            name = header['name'].lower()
            value = header['value']
            
            if name == 'subject':
                parsed['subject'] = value
            elif name == 'from':
                parsed['from'] = value
            elif name == 'to':
                parsed['to'] = value
            elif name == 'cc':
                parsed['cc'] = value
            elif name == 'date':
                parsed['date'] = value
        
        # Extract body
        payload = message.get('payload', {})
        parsed['body_text'], parsed['body_html'] = self._extract_body(payload)
        
        # Extract attachments
        parsed['attachments'] = self._extract_attachments(payload)
        
        return parsed
    
    def _extract_body(self, payload: Dict[str, Any]) -> tuple[str, str]:
        """Extract text and HTML body from message payload"""
        text_body = ""
        html_body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')
                
                if mime_type == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        text_body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                
                elif mime_type == 'text/html':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        html_body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                
                elif 'parts' in part:
                    # Recursive for nested parts
                    nested_text, nested_html = self._extract_body(part)
                    text_body += nested_text
                    html_body += nested_html
        else:
            # No parts, direct body
            data = payload.get('body', {}).get('data', '')
            if data:
                mime_type = payload.get('mimeType', '')
                decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                
                if mime_type == 'text/html':
                    html_body = decoded
                else:
                    text_body = decoded
        
        # If no text but have HTML, extract text from HTML
        if not text_body and html_body:
            text_body = extract_text_from_html(html_body)
        
        return clean_email_text(text_body), html_body
    
    def _extract_attachments(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract attachment metadata from payload"""
        attachments = []
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('filename'):
                    attachment = {
                        'filename': part['filename'],
                        'mime_type': part.get('mimeType', ''),
                        'size': part.get('body', {}).get('size', 0),
                        'attachment_id': part.get('body', {}).get('attachmentId', '')
                    }
                    attachments.append(attachment)
                
                # Recursive for nested parts
                if 'parts' in part:
                    attachments.extend(self._extract_attachments(part))
        
        return attachments
    
    def search_emails(
        self,
        query: Optional[str] = None,
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        after_date: Optional[datetime] = None,
        before_date: Optional[datetime] = None,
        has_attachment: bool = False,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search emails with multiple criteria
        
        Args:
            query: Free text search
            sender: Filter by sender email
            subject: Filter by subject
            after_date: Emails after this date
            before_date: Emails before this date
            has_attachment: Filter emails with attachments
            max_results: Maximum results
            
        Returns:
            List of parsed messages
        """
        # Build Gmail query
        query_parts = []
        
        if query:
            query_parts.append(query)
        
        if sender:
            query_parts.append(f"from:{sender}")
        
        if subject:
            query_parts.append(f"subject:{subject}")
        
        if after_date:
            date_str = after_date.strftime('%Y/%m/%d')
            query_parts.append(f"after:{date_str}")
        
        if before_date:
            date_str = before_date.strftime('%Y/%m/%d')
            query_parts.append(f"before:{date_str}")
        
        if has_attachment:
            query_parts.append("has:attachment")
        
        gmail_query = " ".join(query_parts)
        
        logger.info(f"Searching emails with query: {gmail_query}")
        
        # Get messages
        messages = self.list_messages(query=gmail_query, max_results=max_results)
        
        # Parse messages
        parsed_messages = []
        for msg in messages:
            full_message = self.get_message(msg['id'])
            if full_message:
                parsed = self.parse_message(full_message)
                parsed_messages.append(parsed)
        
        return parsed_messages
    
    def get_recent_emails(self, days: int = 7, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent emails from last N days
        
        Args:
            days: Number of days to look back
            max_results: Maximum results
            
        Returns:
            List of parsed messages
        """
        after_date = datetime.now() - timedelta(days=days)
        return self.search_emails(after_date=after_date, max_results=max_results)

