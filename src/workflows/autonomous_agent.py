"""
Autonomous Email Agent
Proper agentic system with LLM-driven tool selection
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from src.config import get_settings
from src.utils import get_logger, setup_logger
from src.api import AzureOpenAIClient
from src.memory import MemoryManager, VectorStore, EmailStorage
from src.tools import EmailTools, CalendarTools, TaskTools
from src.agents.base_agent import BaseAgent

setup_logger()
logger = get_logger()


class AutonomousEmailAgent:
    """
    Fully autonomous email assistant
    - LLM decides which tools to use
    - Persistent conversation memory
    - Multi-tool execution support
    - No hardcoded routing
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize core components
        self.azure_client = AzureOpenAIClient()
        self.memory_manager = MemoryManager()
        self.vector_store = VectorStore()
        self.email_storage = EmailStorage()  # For full email body retrieval
        
        # Initialize tools
        self.email_tools = EmailTools()
        self.calendar_tools = CalendarTools()
        self.task_tools = TaskTools()
        
        # Initialize base agent
        self.agent = BaseAgent()
        
        # Register all available tools
        self._register_tools()
        
        logger.info("Autonomous Email Agent initialized successfully")
    
    def _register_tools(self):
        """Register all tools with the agent"""
        
        # PRIMARY: RAG-based email search (use this first - emails are indexed daily)
        self.agent.register_tool(
            name="search_emails_rag",
            function=self._search_emails_rag,
            description="PRIMARY tool for email search. Searches indexed emails in vector database with rich metadata filters. Use this FIRST for any email query since emails are indexed daily via autoindex. Supports filtering by sender, date, category, attachments, and semantic meaning.",
            parameters={
                "query": "Semantic search query (e.g., 'placement drives', 'internship opportunities', 'project deadlines')",
                "sender": "Filter by sender email or name (optional, e.g., 'stu.poc@iiitg.ac.in')",
                "date_after": "Filter emails after this date (optional, format: 'YYYY-MM-DD' or 'today', 'yesterday', 'last 3 days')",
                "category": "Filter by email category (optional: 'placement', 'academic', 'meeting', 'notification', etc.)",
                "has_attachments": "Filter by attachment presence (optional: true/false)",
                "n_results": "Number of results to return (default: 10, max: 20)"
            }
        )
        
        # Get full email content by ID (use after search_emails_rag to get details)
        self.agent.register_tool(
            name="get_email_details",
            function=self._get_email_details,
            description="Get full email content and metadata by email_id. Use this AFTER search_emails_rag when you need complete email details. Prevents repeated searches.",
            parameters={
                "email_id": "Email ID from search_emails_rag results (required)"
            }
        )
        
        # FALLBACK: Gmail API search (only if RAG doesn't have the email or need very fresh data)
        self.agent.register_tool(
            name="search_emails_gmail",
            function=self._search_emails_gmail,
            description="FALLBACK tool - Search emails directly from Gmail API. Only use if: (1) search_emails_rag didn't find what you need, (2) you need emails from today that might not be indexed yet, or (3) you need very specific Gmail filters. Otherwise, prefer search_emails_rag.",
            parameters={
                "query": "Gmail search query (Gmail syntax: 'from:email subject:term')",
                "sender": "Filter by sender email (optional)",
                "max_results": "Maximum results (default: 10)"
            }
        )
        
        # Calendar tools
        self.agent.register_tool(
            name="get_upcoming_events",
            function=self._get_upcoming_events,
            description="Get upcoming calendar events",
            parameters={
                "days": "Number of days ahead (default: 7)"
            }
        )
        
        self.agent.register_tool(
            name="create_calendar_event",
            function=self._create_calendar_event,
            description="Create a new calendar event",
            parameters={
                "title": "Event title",
                "start_time": "Start datetime (ISO format or relative like 'tomorrow 10am')",
                "duration_hours": "Duration in hours (default: 1)",
                "description": "Event description (optional)"
            }
        )
        
        # Task tools
        self.agent.register_tool(
            name="create_task",
            function=self._create_task,
            description="Create a new task in Google Tasks",
            parameters={
                "title": "Task title",
                "notes": "Task notes/description (optional)",
                "due_date": "Due date (ISO format or relative like 'tomorrow')"
            }
        )
        
        self.agent.register_tool(
            name="list_tasks",
            function=self._list_tasks,
            description="List all pending tasks",
            parameters={}
        )
        
        # Memory/context tool
        self.agent.register_tool(
            name="recall_conversation",
            function=self._recall_conversation,
            description="Recall previous conversation context. Use when user refers to previous discussion.",
            parameters={
                "query": "What to recall (e.g., 'previous topic', 'last discussion')"
            }
        )
        
        logger.info(f"Registered {len(self.agent.available_tools)} tools")
    
    # Tool implementations
    
    async def _search_emails_rag(
        self,
        query: str,
        sender: str = None,
        date_after: str = None,
        category: str = None,
        has_attachments: bool = None,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        PRIMARY RAG search tool with rich metadata filtering
        This is the main tool since emails are indexed daily via autoindex
        """
        try:
            from datetime import datetime, timedelta
            import re
            
            # Parse date_after if provided
            date_filter = None
            if date_after:
                date_lower = date_after.lower().strip()
                now = datetime.now()
                
                if 'today' in date_lower:
                    date_filter = now.strftime('%Y-%m-%d')
                elif 'yesterday' in date_lower:
                    date_filter = (now - timedelta(days=1)).strftime('%Y-%m-%d')
                elif 'last' in date_lower and 'day' in date_lower:
                    # Extract number of days
                    match = re.search(r'(\d+)', date_lower)
                    if match:
                        days = int(match.group(1))
                        date_filter = (now - timedelta(days=days)).strftime('%Y-%m-%d')
                else:
                    # Try to parse as date
                    try:
                        date_filter = datetime.strptime(date_after, '%Y-%m-%d').strftime('%Y-%m-%d')
                    except:
                        pass
            
            # Build metadata filter
            # ChromaDB requires multiple conditions to be wrapped in $and operator
            conditions = [{"content_type": "email"}]
            
            # Note: ChromaDB doesn't support $contains, so we'll post-filter sender matches
            # Don't add sender to ChromaDB filter - we'll filter results after
            
            if category:
                conditions.append({"category": category})
            
            if has_attachments is not None:
                conditions.append({"has_attachments": has_attachments})
            
            # Build where filter - use $and if multiple conditions, otherwise single condition
            if len(conditions) > 1:
                where_filter = {"$and": conditions}
            else:
                where_filter = conditions[0]
            
            # Perform semantic search with filters
            results = self.vector_store.semantic_search(
                query=query,
                n_results=min(n_results, 20),  # Cap at 20
                filter_metadata=where_filter if where_filter else None
            )
            
            # Process and format results
            formatted_emails = []
            if results.get('ids') and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                    document = results['documents'][0][i] if results.get('documents') and results['documents'][0] else ""
                    distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else 1.0
                    
                    # Post-filter by sender (ChromaDB doesn't support $contains)
                    if sender:
                        email_sender = metadata.get('sender', '').lower()
                        sender_lower = sender.lower()
                        if sender_lower not in email_sender:
                            continue  # Skip if sender doesn't match
                    
                    # Apply date filter in post-processing (ChromaDB date filtering is limited)
                    email_date = metadata.get('date', '')
                    if date_filter and email_date:
                        try:
                            # Parse email date (format varies)
                            email_dt = datetime.fromisoformat(email_date.replace('Z', '+00:00'))
                            filter_dt = datetime.fromisoformat(date_filter)
                            if email_dt.date() < filter_dt.date():
                                continue  # Skip emails before date_filter
                        except:
                            pass  # If date parsing fails, include the email
                    
                    # Extract summary from metadata or document
                    summary = metadata.get('summary', '')
                    if not summary and document:
                        # Use first 200 chars of document as summary
                        summary = document[:200] + "..." if len(document) > 200 else document
                    
                    # CRITICAL: Always return original_email_id, not chunk ID
                    # This ensures agent can retrieve full email from EmailStorage
                    original_email_id = metadata.get('original_email_id')
                    if not original_email_id:
                        original_email_id = metadata.get('email_id', 'unknown')
                    
                    # Remove chunk suffix if present (e.g., "19a97eacb0301e02_chunk_0" -> "19a97eacb0301e02")
                    if '_chunk_' in original_email_id:
                        original_email_id = original_email_id.split('_chunk_')[0]
                    
                    formatted_emails.append({
                        'email_id': original_email_id,  # Always use original email ID
                        'subject': metadata.get('subject', 'N/A'),
                        'sender': metadata.get('sender', 'N/A'),
                        'date': metadata.get('date', 'N/A'),
                        'summary': summary,
                        'category': metadata.get('category', 'N/A'),
                        'has_attachments': metadata.get('has_attachments', False),
                        'attachment_count': metadata.get('attachment_count', 0),
                        'relevance_score': f"{1.0 - distance:.3f}",  # Convert distance to similarity
                        'snippet': document[:300] if document else "",  # First 300 chars
                        'is_chunk': metadata.get('is_chunked', False)  # Indicate if this was from a chunk
                    })
            
            return {
                'emails': formatted_emails,
                'count': len(formatted_emails),
                'query': query,
                'filters_applied': {
                    'sender': sender,
                    'date_after': date_after,
                    'category': category,
                    'has_attachments': has_attachments
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return {'emails': [], 'count': 0, 'error': str(e)}
    
    async def _get_email_details(self, email_id: str) -> Dict[str, Any]:
        """
        Get full email content by ID
        Use this after search_emails_rag to get complete details
        
        Strategy:
        1. Clean email_id (remove chunk suffix if present)
        2. First try EmailStorage (stores full email bodies)
        3. If not found, check if it's a chunk and get original email
        4. Fallback to Gmail API
        """
        try:
            # Step 0: Clean email_id - remove chunk suffix if present
            # e.g., "19a97eacb0301e02_chunk_0" -> "19a97eacb0301e02"
            clean_email_id = email_id
            if '_chunk_' in email_id:
                clean_email_id = email_id.split('_chunk_')[0]
                logger.debug(f"Cleaned chunk ID: {email_id} -> {clean_email_id}")
            
            # Step 1: Try to get from EmailStorage (full email storage)
            stored_email = self.email_storage.get_email(clean_email_id)
            if stored_email:
                body_text = stored_email.get('body_text', '')
                logger.info(f"Retrieved full email from storage: {clean_email_id} (body length: {len(body_text)} chars)")
                return {
                    'email_id': clean_email_id,
                    'subject': stored_email.get('subject', 'N/A'),
                    'sender': stored_email.get('sender', 'N/A'),
                    'recipients': stored_email.get('recipients', 'N/A'),
                    'cc': stored_email.get('cc', ''),
                    'date': stored_email.get('date', 'N/A'),
                    'body': body_text,
                    'body_html': stored_email.get('body_html', ''),
                    'summary': stored_email.get('summary', ''),
                    'category': stored_email.get('category', 'N/A'),
                    'attachments': stored_email.get('attachments', []),
                    'labels': stored_email.get('labels', []),
                    'thread_id': stored_email.get('thread_id', ''),
                    'source': 'email_storage',
                    'complete': True,
                    'body_length': len(body_text),
                    'message': f'Complete email retrieved. Body contains {len(body_text)} characters. This is the FULL email content - no need to fetch again.'
                }
            
            # Step 2: Try with original email_id (if we cleaned a chunk ID)
            if clean_email_id != email_id:
                stored_email = self.email_storage.get_email(clean_email_id)
                if stored_email:
                    logger.debug(f"Retrieved original email after cleaning chunk ID: {clean_email_id}")
                    return {
                        'email_id': clean_email_id,
                        'subject': stored_email.get('subject', 'N/A'),
                        'sender': stored_email.get('sender', 'N/A'),
                        'recipients': stored_email.get('recipients', 'N/A'),
                        'cc': stored_email.get('cc', ''),
                        'date': stored_email.get('date', 'N/A'),
                        'body': stored_email.get('body_text', ''),
                        'body_html': stored_email.get('body_html', ''),
                        'summary': stored_email.get('summary', ''),
                        'category': stored_email.get('category', 'N/A'),
                        'attachments': stored_email.get('attachments', []),
                        'labels': stored_email.get('labels', []),
                        'thread_id': stored_email.get('thread_id', ''),
                        'source': 'email_storage',
                        'complete': True,
                        'note': f'Retrieved original email (was chunk {email_id})'
                    }
            
            # Step 3: Fallback to Gmail API (use clean_email_id, not chunk ID)
            logger.info(f"Email {clean_email_id} not in storage, fetching from Gmail API")
            email_data = self.email_tools.get_email_by_id(clean_email_id)
            if email_data:
                body_text = email_data.get('body_text', email_data.get('body', ''))
                logger.info(f"Retrieved email from Gmail API: {clean_email_id} (body length: {len(body_text)} chars)")
                return {
                    'email_id': clean_email_id,
                    'subject': email_data.get('subject', 'N/A'),
                    'sender': email_data.get('from', 'N/A'),
                    'date': email_data.get('date', 'N/A'),
                    'body': body_text,
                    'body_html': email_data.get('body_html', ''),
                    'attachments': email_data.get('attachments', []),
                    'source': 'gmail_api',
                    'complete': True,
                    'body_length': len(body_text),
                    'message': f'Complete email retrieved from Gmail API. Body contains {len(body_text)} characters. This is the FULL email content - no need to fetch again.'
                }
            
            return {'error': f'Email {clean_email_id} not found in storage or Gmail API'}
            
        except Exception as e:
            logger.error(f"Error getting email details: {e}")
            return {'error': str(e)}
    
    def _search_emails_gmail(
        self,
        query: str = "",
        sender: str = None,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Search emails from Gmail API (live data)"""
        try:
            results = self.email_tools.search_emails(
                query=query,
                sender=sender,
                max_results=max_results
            )
            
            # Format for LLM
            formatted = []
            for email in results[:max_results]:
                formatted.append({
                    'subject': email.get('subject', 'N/A'),
                    'sender': email.get('from', 'N/A'),
                    'date': email.get('date', 'N/A'),
                    'snippet': email.get('snippet', '')[:200]
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error in Gmail search: {e}")
            return []
    
    def _get_recent_emails(self, days: int = 7, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get recent emails"""
        try:
            results = self.email_tools.get_recent_emails(days=days, max_results=max_results)
            
            formatted = []
            for email in results:
                formatted.append({
                    'subject': email.get('subject', 'N/A'),
                    'sender': email.get('from', 'N/A'),
                    'date': email.get('date', 'N/A')
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error getting recent emails: {e}")
            return []
    
    def _get_upcoming_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming calendar events"""
        try:
            events = self.calendar_tools.get_upcoming_events(days=days)
            
            formatted = []
            for event in events:
                start = event.get('start', {})
                formatted.append({
                    'title': event.get('summary', 'N/A'),
                    'start': start.get('dateTime', start.get('date', 'N/A')),
                    'location': event.get('location', '')
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def _create_calendar_event(
        self,
        title: str,
        start_time: str,
        duration_hours: int = 1,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create calendar event"""
        try:
            # Parse start time
            start_dt = self._parse_datetime(start_time)
            end_dt = start_dt + timedelta(hours=duration_hours)
            
            event = self.calendar_tools.create_event(
                title=title,
                start_time=start_dt,
                end_time=end_dt,
                description=description
            )
            
            if event:
                return {
                    'status': 'success',
                    'event_id': event.get('id', ''),
                    'title': title,
                    'start': start_dt.isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Failed to create event'}
                
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _create_task(
        self,
        title: str,
        notes: str = "",
        due_date: str = None
    ) -> Dict[str, Any]:
        """Create a task"""
        try:
            # Parse due date if provided
            due_dt = self._parse_datetime(due_date) if due_date else None
            
            task = self.task_tools.create_task(
                title=title,
                notes=notes,
                due_date=due_dt
            )
            
            if task:
                return {
                    'status': 'success',
                    'task_id': task.get('id', ''),
                    'title': title
                }
            else:
                return {'status': 'error', 'message': 'Failed to create task'}
                
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _list_tasks(self) -> List[Dict[str, Any]]:
        """List pending tasks"""
        try:
            tasks = self.task_tools.get_pending_tasks()
            
            formatted = []
            for task in tasks:
                formatted.append({
                    'title': task.get('title', 'N/A'),
                    'notes': task.get('notes', '')[:100],
                    'due': task.get('due', 'No due date')
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []
    
    async def _recall_conversation(self, query: str) -> str:
        """Recall relevant previous conversation"""
        try:
            # Search conversation memory
            results = self.memory_manager.get_relevant_memory(query, n_results=3)
            
            if not results or not results.get('documents') or not results['documents'][0]:
                return "No relevant previous conversation found."
            
            recalled = []
            for i, doc in enumerate(results['documents'][0][:3]):
                recalled.append(f"Previous discussion {i+1}: {doc[:200]}...")
            
            return "\n".join(recalled)
            
        except Exception as e:
            logger.error(f"Error recalling conversation: {e}")
            return "Could not recall previous conversation."
    
    def _parse_datetime(self, time_str: str) -> datetime:
        """Parse various datetime formats"""
        time_lower = time_str.lower().strip()
        now = datetime.now()
        
        # Relative times
        if 'tomorrow' in time_lower:
            base_date = now + timedelta(days=1)
            # Extract time if specified
            if '10am' in time_lower or '10:00' in time_lower:
                return base_date.replace(hour=10, minute=0, second=0)
            return base_date.replace(hour=9, minute=0, second=0)
        
        elif 'today' in time_lower:
            if '10am' in time_lower:
                return now.replace(hour=10, minute=0, second=0)
            return now
        
        # Try ISO format
        try:
            from dateutil import parser
            return parser.parse(time_str)
        except:
            # Default to tomorrow 9am
            return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0)
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query with full autonomy
        
        Args:
            query: User query
            
        Returns:
            Response with answer and metadata
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Get conversation history (always include current query context)
            conversation_history = self.memory_manager.get_conversation_history()
            
            # Ensure conversation history is available and properly formatted
            if conversation_history:
                logger.debug(f"Using conversation history with {len(conversation_history)} messages")
            else:
                logger.debug("No conversation history available - starting fresh conversation")
            
            # Run autonomous agent
            response = await self.agent.run(
                query=query,
                conversation_history=conversation_history or []
            )
            
            # Store in memory
            await self.memory_manager.learn_from_interaction(
                user_query=query,
                assistant_response=response['answer'],
                context={
                    'tool_calls': len(response.get('tool_calls', [])),
                    'iterations': response.get('iterations', 0)
                }
            )
            
            # Format response
            final_response = {
                'answer': response['answer'],
                'query': query,
                'tool_calls': response.get('tool_calls', []),
                'iterations': response.get('iterations', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Query processing completed successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"I encountered an error: {str(e)}",
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.memory_manager.get_stats()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.memory_manager.clear_conversation()
    
    def clear_email_storage(self):
        """Clear email storage (full email bodies)"""
        self.email_storage.clear_all()
        logger.info("Email storage cleared")

