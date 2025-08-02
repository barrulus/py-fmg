"""Database connection utilities."""

from sqlalchemy import create_engine, text  # ✅ Added text here
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import structlog
from contextlib import contextmanager

from ..config.config import settings
from .models import Base

logger = structlog.get_logger()


class Database:
    """Database connection manager."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None

    def initialize(self):
        """Initialize database connection."""
        logger.info("Initializing database connection", url=settings.database_url)

        # Create engine
        self.engine = create_engine(
            settings.database_url,
            poolclass=StaticPool,
            echo=False,  # Set to True for SQL debugging
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # Create tables
        self.create_tables()

        logger.info("Database connection initialized")

    def create_tables(self):
        """Create all tables."""
        try:
            # Ensure PostGIS extension exists
            from sqlalchemy import text

            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

                conn.commit()

            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created")

        except Exception as e:
            logger.error("Failed to create tables", error=str(e))
            raise

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database instance
db = Database()
