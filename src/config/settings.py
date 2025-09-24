from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "social_support_db"
    postgres_user: str = "postgres"
    postgres_password: str = "password"

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "social_support_docs"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2:7b-chat"

    # Langfuse Configuration
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"

    # Application Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    streamlit_port: int = 8501
    secret_key: str = "dev-secret-key-change-in-production"
    environment: str = "development"

    # File Upload Configuration
    max_file_size: str = "50MB"
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def sqlite_url(self) -> str:
        return "sqlite:///./social_support.db"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()