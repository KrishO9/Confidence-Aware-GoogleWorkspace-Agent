"""
Google Calendar API Client
Handles calendar operations including events, scheduling, etc.
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.config import get_settings
from src.utils import get_logger

logger = get_logger()


class CalendarClient:
    """Client for Google Calendar API operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Calendar API using OAuth2"""
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
        self.service = build('calendar', 'v3', credentials=creds)
        logger.info("Calendar API client authenticated successfully")
    
    def list_events(
        self,
        calendar_id: str = 'primary',
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 100,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List calendar events
        
        Args:
            calendar_id: Calendar ID (default: 'primary')
            time_min: Start time for events
            time_max: End time for events
            max_results: Maximum number of events
            query: Search query
            
        Returns:
            List of events
        """
        try:
            if not self.service:
                logger.error("Calendar service not initialized")
                return []
            
            # Default time range: next 7 days
            if not time_min:
                time_min = datetime.utcnow()
            if not time_max:
                time_max = time_min + timedelta(days=7)
            
            request_params = {
                'calendarId': calendar_id,
                'timeMin': time_min.isoformat() + 'Z',
                'timeMax': time_max.isoformat() + 'Z',
                'maxResults': max_results,
                'singleEvents': True,
                'orderBy': 'startTime'
            }
            
            if query:
                request_params['q'] = query
            
            events_result = self.service.events().list(**request_params).execute()
            events = events_result.get('items', [])
            
            logger.info(f"Found {len(events)} calendar events")
            return events
            
        except HttpError as e:
            logger.error(f"Calendar API error: {e}")
            return []
    
    def get_event(
        self,
        event_id: str,
        calendar_id: str = 'primary'
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific event details
        
        Args:
            event_id: Event ID
            calendar_id: Calendar ID
            
        Returns:
            Event details
        """
        try:
            if not self.service:
                return None
            
            event = self.service.events().get(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()
            
            return event
            
        except HttpError as e:
            logger.error(f"Error getting event {event_id}: {e}")
            return None
    
    def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: str = 'primary'
    ) -> Optional[Dict[str, Any]]:
        """
        Create new calendar event
        
        Args:
            summary: Event title
            start_time: Start datetime
            end_time: End datetime
            description: Event description
            location: Event location
            attendees: List of attendee emails
            calendar_id: Calendar ID
            
        Returns:
            Created event
        """
        try:
            if not self.service:
                return None
            
            event_body = {
                'summary': summary,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC'
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC'
                }
            }
            
            if description:
                event_body['description'] = description
            
            if location:
                event_body['location'] = location
            
            if attendees:
                event_body['attendees'] = [{'email': email} for email in attendees]
            
            event = self.service.events().insert(
                calendarId=calendar_id,
                body=event_body
            ).execute()
            
            logger.info(f"Created event: {event.get('id')}")
            return event
            
        except HttpError as e:
            logger.error(f"Error creating event: {e}")
            return None
    
    def search_events(
        self,
        query: str,
        days_ahead: int = 30,
        calendar_id: str = 'primary'
    ) -> List[Dict[str, Any]]:
        """
        Search events by query
        
        Args:
            query: Search query
            days_ahead: Number of days to search ahead
            calendar_id: Calendar ID
            
        Returns:
            Matching events
        """
        time_min = datetime.utcnow()
        time_max = time_min + timedelta(days=days_ahead)
        
        return self.list_events(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max,
            query=query
        )
    
    def get_upcoming_events(
        self,
        days: int = 7,
        calendar_id: str = 'primary'
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming events
        
        Args:
            days: Number of days ahead
            calendar_id: Calendar ID
            
        Returns:
            List of upcoming events
        """
        time_min = datetime.utcnow()
        time_max = time_min + timedelta(days=days)
        
        return self.list_events(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max
        )

