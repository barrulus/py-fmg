from pathlib import Path
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

import os

# Explicitly load .env for local/dev environments only if values are missing from the environment
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_file = BASE_DIR / ".env"

if env_file.exists():
    file_env = dotenv_values(env_file)
    missing_keys = {k: v for k, v in file_env.items() if k not in os.environ}
    for k, v in missing_keys.items():
        os.environ[k] = v

print(os.environ)

class Settings(BaseSettings):
    """Application settings pulled from environment variables."""

    # Database Configuration
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="py_fmg", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="password", description="Database password")

    @property
    def database_url(self) -> str:
        """Construct full database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # 3D Visualization Configuration
    tiles_output_dir: str = Field(default="./tiles", description="Directory for generated tiles")
    cesium_viewer_port: int = Field(default=8081, description="Cesium viewer port")
    pg2b3dm_port: int = Field(default=8080, description="pg2b3dm service port")

    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for security")
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8081", description="CORS allowed origins")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format (e.g., plain, json)")

    # Map Generation Configuration
    default_map_width: int = Field(default=800, description="Default map width")
    default_map_height: int = Field(default=600, description="Default map height")
    max_map_width: int = Field(default=2000, description="Max allowed map width")
    max_map_height: int = Field(default=2000, description="Max allowed map height")

    # Performance Configuration
    max_concurrent_jobs: int = Field(default=3, description="Max concurrent jobs allowed")
    job_timeout_minutes: int = Field(default=30, description="Job timeout in minutes")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "forbid"  # Explicitly forbid undeclared env vars


# Instantiate singleton settings object
settings = Settings()
