"""Memory and vector database management"""

from .vector_store import VectorStore
from .memory_manager import MemoryManager
from .conversation_memory import ConversationMemory
from .email_storage import EmailStorage

__all__ = [
    "VectorStore",
    "MemoryManager",
    "ConversationMemory"
]

