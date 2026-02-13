"""Configuration management"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    qwen_api_key: str = ""
    
    # Database
    chroma_db_path: str = "./chroma_db"
    
    # Training Configuration
    model_name: str = "Qwen/Qwen3-0.6B"
    device: Literal["cpu", "cuda", "mps"] = "mps"
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_epochs: int = 3
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Local LLM (vLLM / Ollama / llama.cpp server — OpenAI-compatible endpoint)
    local_llm_base_url: str = ""          # e.g. "http://localhost:8080/v1"
    local_llm_model: str = ""             # model name the server expects, e.g. "qwen3-0.6b"
    local_llm_api_key: str = "not-needed" # most local servers don't check this

    # HuggingFace (transformers pipeline, no external server needed)
    hf_model_name: str = ""               # e.g. "Qwen/Qwen3-0.6B"
    hf_device: str = ""                   # "cpu", "cuda", "mps"; empty = auto-detect

    # Kaggle
    kaggle_username: str = ""
    kaggle_key: str = ""
    
    # Logging
    log_level: str = "INFO"
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()


# Create necessary directories
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
