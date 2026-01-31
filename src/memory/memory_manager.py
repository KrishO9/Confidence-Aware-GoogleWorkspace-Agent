"""
Memory Manager
Coordinates between vector store and conversation memory
Implements learning from user behavior
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient
from .vector_store import VectorStore
from .conversation_memory import ConversationMemory

logger = get_logger()


class MemoryManager:
    """Unified memory management for the email assistant"""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.conversation_memory = ConversationMemory()
        self.azure_client = AzureOpenAIClient()
        self.user_preferences: Dict[str, Any] = {}
        self.user_patterns: Dict[str, Any] = {}
        self._load_user_profile()
    
    def _load_user_profile(self):
        """Load user preferences and patterns from disk"""
        try:
            profile_path = Path("data/user_profile.json")
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    self.user_preferences = data.get("preferences", {})
                    self.user_patterns = data.get("patterns", {})
                logger.info("User profile loaded")
            else:
                logger.info("No existing user profile found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
    
    def _save_user_profile(self):
        """Save user preferences and patterns to disk"""
        try:
            profile_path = Path("data/user_profile.json")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "preferences": self.user_preferences,
                "patterns": self.user_patterns,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(profile_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("User profile saved")
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
    
    async def learn_from_interaction(
        self,
        user_query: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Learn from user interaction to improve future responses
        
        Args:
            user_query: User's query
            assistant_response: Assistant's response
            context: Additional context about the interaction
        """
        try:
            # Add to conversation memory ONLY (not to email vector store!)
            self.conversation_memory.add_message("user", user_query)
            self.conversation_memory.add_message("assistant", assistant_response)
            
            # DO NOT store conversations in vector DB during query processing
            # Only extract patterns and preferences
            await self._extract_patterns(user_query, context)
            
            logger.debug(f"Learned from interaction: {user_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    async def _extract_patterns(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Extract patterns from user behavior
        
        Args:
            user_query: User's query
            context: Query context
        """
        try:
            # Update query patterns
            if "query_patterns" not in self.user_patterns:
                self.user_patterns["query_patterns"] = {}
            
            # Simple pattern tracking (can be enhanced with LLM analysis)
            query_lower = user_query.lower()
            
            # Track common keywords
            keywords = ["find", "search", "show", "list", "get", "check", "schedule"]
            for keyword in keywords:
                if keyword in query_lower:
                    count = self.user_patterns["query_patterns"].get(keyword, 0)
                    self.user_patterns["query_patterns"][keyword] = count + 1
            
            # Track time patterns
            now = datetime.now()
            hour = now.hour
            
            if "usage_hours" not in self.user_patterns:
                self.user_patterns["usage_hours"] = {}
            
            hour_count = self.user_patterns["usage_hours"].get(str(hour), 0)
            self.user_patterns["usage_hours"][str(hour)] = hour_count + 1
            
            # Track email preferences if context provided
            if context:
                if "sender" in context:
                    if "frequent_contacts" not in self.user_patterns:
                        self.user_patterns["frequent_contacts"] = {}
                    
                    sender = context["sender"]
                    contact_count = self.user_patterns["frequent_contacts"].get(sender, 0)
                    self.user_patterns["frequent_contacts"][sender] = contact_count + 1
            
            # Save updated profile
            self._save_user_profile()
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
    
    async def get_personalized_context(self, query: str) -> str:
        """
        Get personalized context based on user patterns
        
        Args:
            query: User's current query
            
        Returns:
            Personalized context string
        """
        context_parts = []
        
        # Add relevant preferences
        if self.user_preferences:
            context_parts.append(f"User preferences: {json.dumps(self.user_preferences)}")
        
        # Add relevant patterns
        if "query_patterns" in self.user_patterns:
            top_patterns = sorted(
                self.user_patterns["query_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            if top_patterns:
                patterns_str = ", ".join([f"{k} ({v} times)" for k, v in top_patterns])
                context_parts.append(f"User typically uses: {patterns_str}")
        
        # Add frequent contacts
        if "frequent_contacts" in self.user_patterns:
            top_contacts = sorted(
                self.user_patterns["frequent_contacts"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            if top_contacts:
                contacts_str = ", ".join([k for k, v in top_contacts])
                context_parts.append(f"Frequent contacts: {contacts_str}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def get_relevant_memory(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant memories for query
        
        Args:
            query: User query
            n_results: Number of results
            
        Returns:
            Relevant memories
        """
        return self.vector_store.semantic_search(
            query=query,
            n_results=n_results
        )
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_memory.get_messages()
    
    def clear_conversation(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store": vector_stats,
            "conversation_messages": len(self.conversation_memory.messages),
            "has_summary": self.conversation_memory.summary is not None,
            "user_patterns": len(self.user_patterns),
            "user_preferences": len(self.user_preferences)
        }

