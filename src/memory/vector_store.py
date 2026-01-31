"""
Vector Store using ChromaDB
Persistent storage for embeddings with rich metadata
"""

from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient
import uuid
from datetime import datetime

logger = get_logger()


class VectorStore:
    """Vector database using ChromaDB for semantic search"""
    
    def __init__(self):
        self.settings = get_settings()
        self.azure_client = AzureOpenAIClient()
        self._setup_chroma()
    
    def _setup_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.settings.chroma_collection_name}")
            logger.info(f"Collection contains {self.collection.count()} items")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """
        Ensure collection exists, recreate if needed
        This handles the case where collection was deleted by another instance
        """
        try:
            # Try to access collection
            _ = self.collection.count()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning(f"Collection not found, recreating: {self.settings.chroma_collection_name}")
                try:
                    self.collection = self.chroma_client.get_or_create_collection(
                        name=self.settings.chroma_collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Collection recreated: {self.settings.chroma_collection_name}")
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate collection: {recreate_error}")
                    raise
            else:
                raise
    
    def add_email(
        self,
        email_id: str,
        subject: str,
        body: str,
        sender: str,
        recipients: str,
        date: str,
        labels: List[str],
        attachments: List[Dict[str, str]],
        thread_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add email to vector store
        
        Args:
            email_id: Gmail message ID
            subject: Email subject
            body: Email body text
            sender: Sender email
            recipients: Recipients (comma-separated)
            date: Email date
            labels: Gmail labels
            attachments: List of attachment metadata
            thread_id: Gmail thread ID
            additional_metadata: Additional metadata
            
        Returns:
            Document ID in vector store
        """
        try:
            # Ensure collection exists (handles deleted collections)
            self._ensure_collection_exists()
            
            # Combine text for embedding
            combined_text = f"Subject: {subject}\n\nFrom: {sender}\n\nBody: {body}"
            
            # Generate embedding
            embedding = self.azure_client.generate_embedding(combined_text)
            
            # Prepare metadata
            metadata = {
                "email_id": email_id,
                "subject": subject,
                "sender": sender,
                "recipients": recipients,
                "date": date,
                "labels": ",".join(labels) if labels else "",
                "has_attachments": len(attachments) > 0,
                "attachment_count": len(attachments),
                "thread_id": thread_id or "",
                "indexed_at": datetime.now().isoformat(),
                "content_type": "email"
            }
            
            # Add attachment info
            if attachments:
                attachment_names = [att.get('filename', '') for att in attachments]
                metadata["attachment_names"] = ",".join(attachment_names)
            
            # Merge additional metadata - convert lists to strings for ChromaDB
            if additional_metadata:
                for key, value in additional_metadata.items():
                    if isinstance(value, list):
                        # Convert lists to comma-separated strings
                        metadata[key] = ",".join(str(v) for v in value) if value else ""
                    elif value is None:
                        # ChromaDB doesn't like None values
                        metadata[key] = ""
                    else:
                        metadata[key] = value
            
            # Generate unique document ID
            doc_id = f"email_{email_id}_{uuid.uuid4().hex[:8]}"
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added email to vector store: {email_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding email to vector store: {e}")
            raise
    
    def add_conversation(
        self,
        conversation_text: str,
        user_query: str,
        assistant_response: str,
        timestamp: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add conversation to vector store for memory
        
        Args:
            conversation_text: Full conversation text
            user_query: User's query
            assistant_response: Assistant's response
            timestamp: Conversation timestamp
            additional_metadata: Additional metadata
            
        Returns:
            Document ID
        """
        try:
            # Ensure collection exists
            self._ensure_collection_exists()
            
            # Generate embedding
            embedding = self.azure_client.generate_embedding(conversation_text)
            
            # Prepare metadata
            metadata = {
                "user_query": user_query[:500],  # Limit length
                "assistant_response": assistant_response[:500],
                "timestamp": timestamp,
                "content_type": "conversation",
                "indexed_at": datetime.now().isoformat()
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Generate unique document ID
            doc_id = f"conv_{uuid.uuid4().hex}"
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added conversation to vector store: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding conversation: {e}")
            raise
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Semantic search in vector store
        
        Args:
            query: Search query
            n_results: Number of results
            filter_metadata: Metadata filters (ChromaDB where clause)
            
        Returns:
            Search results with documents, metadatas, distances
        """
        try:
            # Ensure collection exists
            self._ensure_collection_exists()
            
            # Check if collection is empty
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("Collection is empty, returning no results")
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Generate query embedding
            query_embedding = self.azure_client.generate_embedding(query)
            
            # Ensure n_results is at least 1 and not more than collection size
            actual_n_results = max(1, min(n_results, collection_count))
            
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": actual_n_results
            }
            
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            # Search
            results = self.collection.query(**query_params)
            
            logger.info(f"Semantic search returned {len(results['ids'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def search_emails(
        self,
        query: str,
        sender: Optional[str] = None,
        date_after: Optional[str] = None,
        has_attachments: Optional[bool] = None,
        labels: Optional[List[str]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search emails with filters
        
        Args:
            query: Search query
            sender: Filter by sender
            date_after: Filter by date
            has_attachments: Filter by attachment presence
            labels: Filter by labels
            n_results: Number of results
            
        Returns:
            List of matching emails with metadata
        """
        # Build metadata filter
        # ChromaDB requires multiple conditions to be wrapped in $and operator
        conditions = [{"content_type": "email"}]
        
        # Note: ChromaDB doesn't support $contains for partial string matching
        # Sender filtering will be done in post-processing
        
        if has_attachments is not None:
            conditions.append({"has_attachments": has_attachments})
        
        # Build where filter - use $and if multiple conditions, otherwise single condition
        if len(conditions) > 1:
            where_filter = {"$and": conditions}
        else:
            where_filter = conditions[0]
        
        # Note: ChromaDB has limited filtering, complex filters may need post-processing
        
        results = self.semantic_search(
            query=query,
            n_results=n_results,
            filter_metadata=where_filter
        )
        
        # Process results with post-filtering
        emails = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                
                # Post-filter by sender (ChromaDB doesn't support $contains)
                if sender:
                    email_sender = metadata.get('sender', '').lower()
                    sender_lower = sender.lower()
                    if sender_lower not in email_sender:
                        continue  # Skip if sender doesn't match
                
                # Always use original_email_id if available (for chunked emails)
                email_id = metadata.get('original_email_id') or metadata.get('email_id', 'unknown')
                
                email_data = {
                    'id': email_id,  # Use original email ID, not chunk ID
                    'content': results['documents'][0][i],
                    'metadata': metadata,
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                emails.append(email_data)
        
        return emails
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data
        """
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_by_id(self, doc_id: str):
        """Delete document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.settings.chroma_collection_name,
            "persist_directory": self.settings.chroma_persist_directory
        }
    
    def clear_all_data(self):
        """Clear all data from vector store"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name=self.settings.chroma_collection_name)
            logger.info(f"Deleted collection: {self.settings.chroma_collection_name}")
            
            # Recreate the collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Recreated collection: {self.settings.chroma_collection_name}")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise

