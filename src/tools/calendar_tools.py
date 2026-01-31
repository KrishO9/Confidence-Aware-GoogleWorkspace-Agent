"""
Calendar Tools
Tools for calendar operations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from src.api import CalendarClient
from src.utils import get_logger

logger = get_logger()


class CalendarTools:
    """Calendar operation tools for agents"""
    
    def __init__(self):
        self.calendar_client = CalendarClient()
    
    def get_upcoming_events(
        self,
        days: int = 7,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming calendar events
        
        Args:
            days: Number of days ahead
            max_results: Maximum results
            
        Returns:
            List of upcoming events
        """
        try:
            events = self.calendar_client.get_upcoming_events(
                days=days,
                calendar_id='primary'
            )
            
            return events[:max_results]
            
        except Exception as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []
    
    def search_events(
        self,
        query: str,
        days_ahead: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Search calendar events
        
        Args:
            query: Search query
            days_ahead: Days to search ahead
            
        Returns:
            Matching events
        """
        try:
            events = self.calendar_client.search_events(
                query=query,
                days_ahead=days_ahead
            )
            
            return events
            
        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return []
    
    def create_event(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create new calendar event
        
        Args:
            title: Event title
            start_time: Start datetime
            end_time: End datetime
            description: Event description
            location: Event location
            
        Returns:
            Created event
        """
        try:
            event = self.calendar_client.create_event(
                summary=title,
                start_time=start_time,
                end_time=end_time,
                description=description,
                location=location
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return None
    
    def get_today_events(self) -> List[Dict[str, Any]]:
        """Get today's events"""
        return self.get_upcoming_events(days=1)
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get tool descriptions for agent"""
        return [
            {
                "name": "get_upcoming_events",
                "description": "Get upcoming calendar events for the next N days",
                "parameters": ["days", "max_results"]
            },
            {
                "name": "search_events",
                "description": "Search calendar events by query",
                "parameters": ["query", "days_ahead"]
            },
            {
                "name": "create_event",
                "description": "Create a new calendar event",
                "parameters": ["title", "start_time", "end_time", "description", "location"]
            },
            {
                "name": "get_today_events",
                "description": "Get today's calendar events",
                "parameters": []
            }
        ]

