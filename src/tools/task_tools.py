"""
Task Tools
Tools for task management operations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from src.api import TasksClient
from src.utils import get_logger

logger = get_logger()


class TaskTools:
    """Task management tools for agents"""
    
    def __init__(self):
        self.tasks_client = TasksClient()
    
    def list_tasks(
        self,
        show_completed: bool = False,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List tasks
        
        Args:
            show_completed: Include completed tasks
            max_results: Maximum results
            
        Returns:
            List of tasks
        """
        try:
            tasks = self.tasks_client.list_tasks(
                show_completed=show_completed,
                max_results=max_results
            )
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []
    
    def create_task(
        self,
        title: str,
        notes: Optional[str] = None,
        due_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create new task
        
        Args:
            title: Task title
            notes: Task notes
            due_date: Due date
            
        Returns:
            Created task
        """
        try:
            task = self.tasks_client.create_task(
                title=title,
                notes=notes,
                due_date=due_date
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None
    
    def complete_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark task as completed
        
        Args:
            task_id: Task ID
            
        Returns:
            Updated task
        """
        try:
            task = self.tasks_client.complete_task(task_id=task_id)
            return task
            
        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return None
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get all pending tasks"""
        return self.list_tasks(show_completed=False)
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get tool descriptions for agent"""
        return [
            {
                "name": "list_tasks",
                "description": "List all tasks, optionally including completed ones",
                "parameters": ["show_completed", "max_results"]
            },
            {
                "name": "create_task",
                "description": "Create a new task with title, notes, and optional due date",
                "parameters": ["title", "notes", "due_date"]
            },
            {
                "name": "complete_task",
                "description": "Mark a task as completed",
                "parameters": ["task_id"]
            },
            {
                "name": "get_pending_tasks",
                "description": "Get all pending (not completed) tasks",
                "parameters": []
            }
        ]

