"""
Azure OpenAI Client
Handles LLM and embedding operations using Azure OpenAI
"""

from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import get_settings
from src.utils import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger()


class AzureOpenAIClient:
    """Client for Azure OpenAI operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize Azure OpenAI clients"""
        base_endpoint = self.settings.azure_openai_base_endpoint
        
        # LangChain Azure OpenAI Chat Model
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=base_endpoint,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            deployment_name=self.settings.azure_openai_deployment,
            max_tokens=4000,
            model_kwargs={
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        )
        
        # LangChain Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=base_endpoint,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            deployment=self.settings.azure_openai_embedding_deployment,
            chunk_size=self.settings.embedding_batch_size
        )
        
        # Direct OpenAI client for advanced operations
        self.client = AzureOpenAI(
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=base_endpoint
        )
        
        logger.info("Azure OpenAI clients initialized successfully")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4000
    ) -> str:
        """
        Generate response from Azure OpenAI
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.settings.azure_openai_deployment,
                messages=messages,
                max_tokens=max_tokens,
                top_p=0.95
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Generated response: {content[:100]}...")
            return content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Use LangChain embeddings
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def get_chat_model(self) -> AzureChatOpenAI:
        """Get LangChain chat model instance"""
        return self.chat_model
    
    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        """Get LangChain embeddings instance"""
        return self.embeddings
    
    async def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate structured output based on JSON schema
        
        Args:
            prompt: Input prompt
            schema: JSON schema for output
            
        Returns:
            Structured output matching schema
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides structured outputs in JSON format."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nProvide output in JSON format matching this schema: {schema}"
                }
            ]
            
            response = await self.generate_response(
                messages=messages,
                max_tokens=2000
            )
            
            # Parse JSON response
            import json
            structured_output = json.loads(response)
            return structured_output
            
        except Exception as e:
            logger.error(f"Error generating structured output: {e}")
            raise

