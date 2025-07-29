"""Configuration management."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Database
    db_user: str = Field(..., description="Database user")
    db_host: str = Field(..., description="Database host")
    db_name: str = Field(..., description="Database name")
    db_password: str = Field(..., description="Database password")
    db_port: int = Field(..., description="Database port")

    @property
    def database_url(self) -> str:
        """Build database URL from components."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Generation
    max_map_size: int = Field(default=2048, description="Maximum map size")
    generation_timeout: int = Field(
        default=300, description="Generation timeout in seconds"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore[call-arg]
