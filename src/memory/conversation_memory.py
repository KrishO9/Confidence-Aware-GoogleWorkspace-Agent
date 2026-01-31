"""
Conversation Memory Management
Handles short-term conversation context with summarization
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient

logger = get_logger()


class ConversationMemory:
    """Manages conversation history with automatic summarization"""
    
    def __init__(self):
        self.settings = get_settings()
        self.azure_client = AzureOpenAIClient()
        self.messages: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
        self.message_count = 0
    
    def add_message(self, role: str, content: str):
        """
        Add message to conversation history
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.messages.append(message)
        self.message_count += 1
        
        # Check if summarization is needed
        if self.message_count >= self.settings.summarization_threshold:
            self._summarize_and_compress()
    
    def get_messages(self, include_summary: bool = True) -> List[Dict[str, str]]:
        """
        Get conversation messages
        
        Args:
            include_summary: Include summary as first message
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add summary if available
        if include_summary and self.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        # Add recent messages
        messages.extend(self.messages)
        
        return messages
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """
        Get N most recent messages
        
        Args:
            n: Number of messages
            
        Returns:
            Recent messages
        """
        return self.messages[-n:] if self.messages else []
    
    async def _summarize_and_compress(self):
        """Summarize old messages and keep recent ones"""
        try:
            if len(self.messages) < self.settings.summarization_threshold:
                return
            
            # Split messages: old to summarize, recent to keep
            split_point = len(self.messages) - 10  # Keep last 10 messages
            old_messages = self.messages[:split_point]
            recent_messages = self.messages[split_point:]
            
            # Build conversation text
            conversation_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in old_messages
            ])
            
            # Generate summary
            summary_prompt = f"""Summarize the following conversation, focusing on:
1. Key topics discussed
2. Important information shared
3. User preferences or patterns
4. Any pending tasks or follow-ups

Conversation:
{conversation_text}

Provide a concise summary (max 500 words):"""
            
            messages_for_llm = [
                {"role": "system", "content": "You are a helpful assistant that summarizes conversations concisely."},
                {"role": "user", "content": summary_prompt}
            ]
            
            new_summary = await self.azure_client.generate_response(
                messages=messages_for_llm,
                max_tokens=1000
            )
            
            # Update memory
            if self.summary:
                # Combine with existing summary
                combined_prompt = f"""Previous summary: {self.summary}

New conversation summary: {new_summary}

Provide a unified summary that combines both:"""
                
                messages_for_llm = [
                    {"role": "system", "content": "You are a helpful assistant that creates unified summaries."},
                    {"role": "user", "content": combined_prompt}
                ]
                
                self.summary = await self.azure_client.generate_response(
                    messages=messages_for_llm,
                    max_tokens=1000
                )
            else:
                self.summary = new_summary
            
            # Keep only recent messages
            self.messages = recent_messages
            self.message_count = len(self.messages)
            
            logger.info("Conversation summarized and compressed")
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.summary = None
        self.message_count = 0
        logger.info("Conversation memory cleared")
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export conversation for storage"""
        return {
            "messages": self.messages,
            "summary": self.summary,
            "message_count": self.message_count,
            "exported_at": datetime.now().isoformat()
        }
    
    def import_conversation(self, data: Dict[str, Any]):
        """Import conversation from storage"""
        self.messages = data.get("messages", [])
        self.summary = data.get("summary")
        self.message_count = data.get("message_count", len(self.messages))
        logger.info("Conversation memory imported")

