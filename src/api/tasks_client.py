"""
Google Tasks API Client
Handles task management operations
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.config import get_settings
from src.utils import get_logger

logger = get_logger()


class TasksClient:
    """Client for Google Tasks API operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Tasks API using OAuth2"""
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
        self.service = build('tasks', 'v1', credentials=creds)
        logger.info("Tasks API client authenticated successfully")
    
    def list_task_lists(self) -> List[Dict[str, Any]]:
        """
        List all task lists
        
        Returns:
            List of task lists
        """
        try:
            if not self.service:
                logger.error("Tasks service not initialized")
                return []
            
            results = self.service.tasklists().list().execute()
            task_lists = results.get('items', [])
            
            logger.info(f"Found {len(task_lists)} task lists")
            return task_lists
            
        except HttpError as e:
            logger.error(f"Tasks API error: {e}")
            return []
    
    def list_tasks(
        self,
        task_list_id: Optional[str] = None,
        show_completed: bool = False,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List tasks from a task list
        
        Args:
            task_list_id: Task list ID (default: first list)
            show_completed: Include completed tasks
            max_results: Maximum results
            
        Returns:
            List of tasks
        """
        try:
            if not self.service:
                return []
            
            # Get default task list if not specified
            if not task_list_id:
                task_lists = self.list_task_lists()
                if not task_lists:
                    return []
                task_list_id = task_lists[0]['id']
            
            results = self.service.tasks().list(
                tasklist=task_list_id,
                showCompleted=show_completed,
                maxResults=max_results
            ).execute()
            
            tasks = results.get('items', [])
            logger.info(f"Found {len(tasks)} tasks")
            return tasks
            
        except HttpError as e:
            logger.error(f"Error listing tasks: {e}")
            return []
    
    def get_task(
        self,
        task_id: str,
        task_list_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific task
        
        Args:
            task_id: Task ID
            task_list_id: Task list ID
            
        Returns:
            Task details
        """
        try:
            if not self.service:
                return None
            
            if not task_list_id:
                task_lists = self.list_task_lists()
                if not task_lists:
                    return None
                task_list_id = task_lists[0]['id']
            
            task = self.service.tasks().get(
                tasklist=task_list_id,
                task=task_id
            ).execute()
            
            return task
            
        except HttpError as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None
    
    def create_task(
        self,
        title: str,
        notes: Optional[str] = None,
        due_date: Optional[datetime] = None,
        task_list_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create new task
        
        Args:
            title: Task title
            notes: Task notes/description
            due_date: Due date
            task_list_id: Task list ID
            
        Returns:
            Created task
        """
        try:
            if not self.service:
                return None
            
            if not task_list_id:
                task_lists = self.list_task_lists()
                if not task_lists:
                    return None
                task_list_id = task_lists[0]['id']
            
            task_body = {'title': title}
            
            if notes:
                task_body['notes'] = notes
            
            if due_date:
                task_body['due'] = due_date.isoformat() + 'Z'
            
            task = self.service.tasks().insert(
                tasklist=task_list_id,
                body=task_body
            ).execute()
            
            logger.info(f"Created task: {task.get('id')}")
            return task
            
        except HttpError as e:
            logger.error(f"Error creating task: {e}")
            return None
    
    def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        notes: Optional[str] = None,
        status: Optional[str] = None,
        task_list_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update existing task
        
        Args:
            task_id: Task ID
            title: New title
            notes: New notes
            status: New status ('needsAction' or 'completed')
            task_list_id: Task list ID
            
        Returns:
            Updated task
        """
        try:
            if not self.service:
                return None
            
            if not task_list_id:
                task_lists = self.list_task_lists()
                if not task_lists:
                    return None
                task_list_id = task_lists[0]['id']
            
            # Get current task
            task = self.get_task(task_id, task_list_id)
            if not task:
                return None
            
            # Update fields
            if title:
                task['title'] = title
            if notes:
                task['notes'] = notes
            if status:
                task['status'] = status
            
            updated_task = self.service.tasks().update(
                tasklist=task_list_id,
                task=task_id,
                body=task
            ).execute()
            
            logger.info(f"Updated task: {task_id}")
            return updated_task
            
        except HttpError as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return None
    
    def complete_task(
        self,
        task_id: str,
        task_list_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Mark task as completed
        
        Args:
            task_id: Task ID
            task_list_id: Task list ID
            
        Returns:
            Updated task
        """
        return self.update_task(
            task_id=task_id,
            status='completed',
            task_list_id=task_list_id
        )

