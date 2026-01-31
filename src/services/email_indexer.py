"""
Automated Email Indexing Service
Periodically pulls and indexes emails with rich metadata
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
from pathlib import Path

from src.config import get_settings
from src.utils import get_logger
from src.api import GmailClient, AzureOpenAIClient
from src.memory import VectorStore, EmailStorage
from src.utils import chunk_text

logger = get_logger()


class EmailIndexerService:
    """
    Background service for automated email indexing
    Pulls emails periodically, extracts content, and stores with rich metadata
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.gmail_client = GmailClient()
        self.azure_client = AzureOpenAIClient()
        self.vector_store = VectorStore()
        self.email_storage = EmailStorage()  # Store full email bodies
        self.is_running = False
        self.last_index_time = None
        self._load_state()
    
    def _load_state(self):
        """Load last indexing state"""
        try:
            state_file = Path("data/indexer_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.last_index_time = datetime.fromisoformat(state.get('last_index_time', ''))
                    logger.info(f"Loaded indexer state: last run at {self.last_index_time}")
        except Exception as e:
            logger.warning(f"Could not load indexer state: {e}")
            self.last_index_time = None
    
    def _save_state(self):
        """Save indexing state"""
        try:
            state_file = Path("data/indexer_state.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'last_index_time': self.last_index_time.isoformat() if self.last_index_time else None,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving indexer state: {e}")
    
    async def start(self):
        """Start the background indexing service"""
        if self.is_running:
            logger.warning("Indexer service is already running")
            return
        
        self.is_running = True
        logger.info("Starting email indexer service")
        
        # Run initial indexing
        await self.run_indexing()
        
        # Schedule periodic indexing
        asyncio.create_task(self._scheduled_indexing())
    
    async def stop(self):
        """Stop the indexing service"""
        self.is_running = False
        logger.info("Stopping email indexer service")
    
    async def _scheduled_indexing(self):
        """Run indexing on schedule"""
        while self.is_running:
            try:
                # Wait for configured interval
                interval_seconds = self.settings.auto_index_interval_hours * 3600
                await asyncio.sleep(interval_seconds)
                
                if self.is_running:
                    await self.run_indexing()
                    
            except Exception as e:
                logger.error(f"Error in scheduled indexing: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def run_indexing(self) -> Dict[str, Any]:
        """
        Run email indexing process
        
        Returns:
            Statistics about indexed emails
        """
        try:
            start_time = datetime.now()
            logger.info("=" * 80)
            logger.info(f"Starting scheduled email indexing at {start_time}")
            logger.info("=" * 80)
            
            # Calculate date range
            days_back = self.settings.auto_index_days_back
            after_date = datetime.now() - timedelta(days=days_back)
            
            # Build Gmail query for recent emails
            gmail_query = f"after:{after_date.strftime('%Y/%m/%d')}"
            
            logger.info(f"Fetching emails from last {days_back} days")
            logger.info(f"Gmail query: {gmail_query}")
            
            # Fetch emails
            messages = self.gmail_client.list_messages(
                query=gmail_query,
                max_results=self.settings.auto_index_max_emails
            )
            
            logger.info(f"Found {len(messages)} emails to process")
            
            indexed_count = 0
            skipped_count = 0
            error_count = 0
            
            for i, msg in enumerate(messages, 1):
                try:
                    # Get full message
                    full_message = self.gmail_client.get_message(msg['id'])
                    if not full_message:
                        skipped_count += 1
                        continue
                    
                    # Parse message
                    parsed = self.gmail_client.parse_message(full_message)
                    
                    # Check if already indexed by searching for exact email_id in metadata
                    # Note: This is a simple check - in production, use a dedicated tracking table
                    try:
                        existing = self.vector_store.collection.get(
                            where={"email_id": parsed['id']},
                            limit=1
                        )
                        
                        if existing and existing['ids']:
                            skipped_count += 1
                            logger.debug(f"Skipping already indexed email: {parsed['id']}")
                            continue
                    except Exception as e:
                        # If check fails, proceed with indexing (better to duplicate than skip)
                        logger.debug(f"Could not check if email exists, proceeding with indexing: {e}")
                    
                    # Generate summary for the email
                    summary = await self._generate_email_summary(parsed)
                    
                    # Store full email content separately (for complete retrieval)
                    parsed['summary'] = summary
                    parsed['category'] = self._categorize_email(
                        parsed.get('subject', ''),
                        parsed.get('body_text', ''),
                        parsed.get('labels', [])
                    )
                    parsed['stored_at'] = datetime.now().isoformat()
                    self.email_storage.store_email(parsed['id'], parsed)
                    
                    # Index email with enhanced metadata (for semantic search)
                    await self._index_email_with_metadata(parsed, summary)
                    
                    indexed_count += 1
                    
                    # Progress logging
                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{len(messages)} processed, {indexed_count} indexed")
                    
                    # Rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Error indexing email {msg.get('id', 'unknown')}: {e}")
                    error_count += 1
            
            # Update state
            self.last_index_time = datetime.now()
            self._save_state()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            stats = {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'total_fetched': len(messages),
                'indexed': indexed_count,
                'skipped': skipped_count,
                'errors': error_count,
                'days_back': days_back
            }
            
            logger.info("=" * 80)
            logger.info("Email indexing completed")
            logger.info(f"  - Total fetched: {stats['total_fetched']}")
            logger.info(f"  - Newly indexed: {stats['indexed']}")
            logger.info(f"  - Skipped (already indexed): {stats['skipped']}")
            logger.info(f"  - Errors: {stats['errors']}")
            logger.info(f"  - Duration: {duration:.2f} seconds")
            logger.info("=" * 80)
            
            return stats
            
        except Exception as e:
            logger.error(f"Critical error in email indexing: {e}")
            raise
    
    async def _generate_email_summary(self, parsed_email: Dict[str, Any]) -> str:
        """
        Generate AI summary of email content
        
        Args:
            parsed_email: Parsed email dictionary
            
        Returns:
            Email summary
        """
        try:
            subject = parsed_email.get('subject', '')
            body = parsed_email.get('body_text', '')[:2000]  # Limit body length
            
            if not body:
                return f"Email about: {subject}"
            
            summary_prompt = f"""Summarize this email concisely (max 3 sentences):

Subject: {subject}

Body:
{body}

Summary:"""
            
            messages = [
                {"role": "system", "content": "You are an email summarization assistant. Create brief, informative summaries."},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary = await self.azure_client.generate_response(
                messages=messages,
                max_tokens=150
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating email summary: {e}")
            return f"Email about: {parsed_email.get('subject', 'Unknown')}"
    
    async def _index_email_with_metadata(
        self,
        parsed_email: Dict[str, Any],
        summary: str
    ):
        """
        Index email with rich metadata structure
        
        Args:
            parsed_email: Parsed email dictionary
            summary: AI-generated summary
        """
        try:
            email_id = parsed_email['id']
            subject = parsed_email.get('subject', '')
            body = parsed_email.get('body_text', '')
            sender = parsed_email.get('from', '')
            recipients = parsed_email.get('to', '')
            date = parsed_email.get('date', '')
            labels = parsed_email.get('labels', [])
            attachments = parsed_email.get('attachments', [])
            thread_id = parsed_email.get('thread_id', '')
            
            # Enhanced metadata structure - all values must be str, int, float, or bool
            # Extract attachment types and convert to comma-separated string
            attachment_types = list(set([att.get('mime_type', '').split('/')[0] for att in attachments if att.get('mime_type')]))
            
            # Enhanced metadata structure - comprehensive email information
            enhanced_metadata = {
                'summary': summary or "",
                'has_attachment': len(attachments) > 0,
                'attachment_count': len(attachments),
                'attachment_types': ",".join(attachment_types) if attachment_types else "",
                'word_count': len(body.split()) if body else 0,
                'char_count': len(body) if body else 0,
                'is_reply': 'Re:' in subject or 'RE:' in subject,
                'is_forward': 'Fwd:' in subject or 'FWD:' in subject,
                'priority': 'IMPORTANT' in labels or 'STARRED' in labels,
                'category': self._categorize_email(subject, body, labels),
                'has_html': bool(parsed_email.get('body_html', '')),
                'sender_domain': sender.split('@')[1] if '@' in sender else '',
                'is_chunked': False,  # Will be set to True for chunks
                'original_email_id': email_id  # For chunked emails, reference original
            }
            
            # Chunk long emails
            if len(body) > 1500:
                chunks = chunk_text(body, chunk_size=1200, overlap=150)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **enhanced_metadata,
                        'is_chunked': True,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_email_id': email_id,
                        'chunk_word_count': len(chunk.split())
                    }
                    self.vector_store.add_email(
                        email_id=f"{email_id}_chunk_{i}",
                        subject=subject,
                        body=chunk,
                        sender=sender,
                        recipients=recipients,
                        date=date,
                        labels=labels,
                        attachments=attachments,
                        thread_id=thread_id,
                        additional_metadata=chunk_metadata
                    )
            else:
                # Index full email
                self.vector_store.add_email(
                    email_id=email_id,
                    subject=subject,
                    body=body,
                    sender=sender,
                    recipients=recipients,
                    date=date,
                    labels=labels,
                    attachments=attachments,
                    thread_id=thread_id,
                    additional_metadata=enhanced_metadata
                )
            
            logger.debug(f"Successfully indexed email: {email_id} - {subject[:50]}")
            
        except Exception as e:
            logger.error(f"Error indexing email with metadata: {e}")
            raise
    
    def _categorize_email(self, subject: str, body: str, labels: List[str]) -> str:
        """
        Categorize email based on content
        
        Args:
            subject: Email subject
            body: Email body
            labels: Gmail labels
            
        Returns:
            Category string
        """
        text = (subject + " " + body).lower()
        
        # Define category keywords
        categories = {
            'placement': ['placement', 'campus', 'recruitment', 'job', 'interview', 'company visit', 'hiring'],
            'academic': ['exam', 'assignment', 'class', 'lecture', 'course', 'semester', 'grade'],
            'meeting': ['meeting', 'schedule', 'calendar', 'conference', 'zoom', 'teams'],
            'notification': ['notification', 'alert', 'reminder', 'update', 'announcement'],
            'personal': ['personal', 'family', 'friend'],
            'work': ['project', 'deadline', 'task', 'report', 'work', 'office']
        }
        
        # Check keywords
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        # Check labels
        if 'CATEGORY_PROMOTIONS' in labels:
            return 'promotional'
        elif 'CATEGORY_SOCIAL' in labels:
            return 'social'
        elif 'CATEGORY_UPDATES' in labels:
            return 'updates'
        
        return 'general'
    
    def get_status(self) -> Dict[str, Any]:
        """Get indexer service status"""
        return {
            'is_running': self.is_running,
            'last_index_time': self.last_index_time.isoformat() if self.last_index_time else None,
            'next_index_time': (
                self.last_index_time + timedelta(hours=self.settings.auto_index_interval_hours)
            ).isoformat() if self.last_index_time else None,
            'config': {
                'interval_hours': self.settings.auto_index_interval_hours,
                'days_back': self.settings.auto_index_days_back,
                'max_emails': self.settings.auto_index_max_emails
            }
        }

