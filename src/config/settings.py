"""
Settings and Configuration Management
Handles all environment variables and configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables (.env). No secrets in code."""

    # Azure OpenAI (set in .env - never commit real values)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""  # Base URL e.g. https://your-resource.openai.azure.com
    azure_openai_deployment: str = "gpt-4"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2025-01-01-preview"

    # Email Indexing Configuration
    auto_index_enabled: bool = True
    auto_index_interval_hours: int = 6
    auto_index_days_back: int = 7
    auto_index_max_emails: int = 500

    # Extract base endpoint (handles full URL or base URL from env)
    @property
    def azure_openai_base_endpoint(self) -> str:
        """Extract base endpoint from env URL (full or base)."""
        if not self.azure_openai_endpoint:
            return ""
        if "/openai/deployments/" in self.azure_openai_endpoint:
            return self.azure_openai_endpoint.split("/openai/deployments/")[0]
        return self.azure_openai_endpoint.split("?")[0].rstrip("/")
    
    # Google Workspace APIs
    google_credentials_path: str = "credentials.json"
    google_token_path: str = "token.json"
    google_scopes: list[str] = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/tasks'
    ]
    
    # Vector Database
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "email_assistant_memory"
    
    # Memory Configuration
    max_conversation_history: int = 50
    summarization_threshold: int = 20
    embedding_batch_size: int = 100
    
    # Search Configuration
    top_k_results: int = 10
    similarity_threshold: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/email_assistant.log"
    
    # Agent Configuration
    max_iterations: int = 15
    agent_timeout: int = 300  # seconds
    parallel_execution_enabled: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.ensure_directories()
    return settings

