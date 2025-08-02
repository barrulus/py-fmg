"""
Database utilities and models.

This package provides:
- SQLAlchemy models for PostGIS schema
- Database connection management
- Export pipeline to PostGIS
- High-performance spatial queries
"""

from .connection import Database, db
from .export import PostGISExporter, export_map_to_postgis
from .queries import PostGISQueries, create_query_helper
from .models import (
    Base, Map, Culture, Religion, State, Settlement, River, BiomeRegion,
    ClimateData, CellCulture, CellReligion, ReligionCulture, GenerationJob
)

__all__ = [
    # Connection management
    'Database', 'db',
    
    # Export functionality
    'PostGISExporter', 'export_map_to_postgis',
    
    # Query functionality
    'PostGISQueries', 'create_query_helper',
    
    # Models
    'Base', 'Map', 'Culture', 'Religion', 'State', 'Settlement', 'River', 
    'BiomeRegion', 'ClimateData', 'CellCulture', 'CellReligion', 
    'ReligionCulture', 'GenerationJob'
]