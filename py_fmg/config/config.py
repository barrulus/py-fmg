"""Configuration management."""


import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Explicitly load .env for local/dev environments (works on Windows, Linux, Docker)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

for key in ["DB_USER", "DB_PASSWORD", "DB_PORT", "DB_HOST"]:
    print(f"{key} = {os.getenv(key)}")

print(BASE_DIR)


class Settings(BaseSettings):
    """Application settings."""

    # Database

    db_user: str = Field(default="user", description="Database user")
    db_host: str = Field(default="localhost", description="Database host")
    db_name: str = Field(default="py-fmg", description="Database name")
    db_password: str = Field(default="password", description="Database password")
    db_port: str = Field(default="5432", description="Database port")  # ⚠️ int default


    @property
    def database_url(self) -> str:
        """Build database URL from components."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Generation
    max_map_size: int = Field(default=2048, description="Maximum map size")

    generation_timeout: int = Field(default=300, description="Generation timeout in seconds")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    class Config:
        env_file = ".env"           # used by Pydantic
        env_file_encoding = "utf-8"

# Instantiate once, to be imported throughout app
settings = Settings()

