#!/usr/bin/env python3
"""Initialize the database for py-fmg."""

from ..config.config import settings
from .connection import db


def main():
    """Initialize the database."""
    try:
        print("Initializing database...")
        db.initialize()
        print("✓ Database initialized successfully!")
        print("✓ Tables created")
        print("✓ PostGIS extension enabled")

    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()

