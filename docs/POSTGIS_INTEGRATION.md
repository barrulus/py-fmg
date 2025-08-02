# PostGIS Integration Guide

This document explains how to use the PostGIS database integration for storing and querying Fantasy Map Generator data.

## Overview

The PostGIS integration provides:
- **Complete database schema** following PLAN.md specifications
- **Export pipeline** from generated map data to PostGIS
- **High-performance spatial queries** for RPG game engine
- **Bulk insert optimization** for large datasets
- **Spatial indexing** for sub-50ms queries

## Database Schema

The schema implements all tables from PLAN.md:

### Level 1: Macro (FMG) Tables

| Table | Description | Geometry Type |
|-------|-------------|---------------|
| `maps` | Map metadata and generation parameters | - |
| `cultures` | Cultural territories and expansion | MULTIPOLYGON |
| `religions` | Religious territories and beliefs | MULTIPOLYGON |
| `states` | Political boundaries and governance | POLYGON |
| `settlements` | Cities, towns, villages | POINT |
| `rivers` | Waterway systems | LINESTRING |
| `biomes` | Ecological regions | MULTIPOLYGON |
| `climate_data` | Temperature and precipitation per cell | - |

## Quick Start

### 1. Database Setup

```bash
# Install PostgreSQL with PostGIS
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib postgis

# macOS:
brew install postgresql postgis
```

### 2. Environment Configuration

Create a `.env` file:

```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fmg_maps
```

### 3. Initialize Database

```python
from py_fmg.db import db

# Initialize connection and create tables
db.initialize()
```

### 4. Export Generated Map

```python
from py_fmg.db import export_map_to_postgis

# After generating map data
with db.get_session() as session:
    map_id = export_map_to_postgis(
        session=session,
        map_name="My Fantasy World",
        map_data={
            'metadata': generation_params,
            'voronoi_graph': voronoi_graph,
            'cultures': cultures_dict,
            'religions': religions_dict,
            'states': states_dict,
            'settlements': settlements_dict,
            'rivers': rivers_dict,
            'biomes': biomes_dict,
            'climate': climate_data
        },
        bulk_insert=True  # Use bulk optimization
    )
    print(f"Map exported: {map_id}")
```

## Export Pipeline Features

### Automatic Geometry Creation

The export pipeline automatically converts FMG data structures into proper PostGIS geometries:

- **Territorial Polygons**: Cultures, religions, states, and biomes get MULTIPOLYGON geometries from their assigned cells
- **Settlement Points**: Cities and towns become POINT geometries at their cell coordinates  
- **River Networks**: Rivers become LINESTRING geometries following their cell paths
- **Spatial Indexing**: GIST indexes created automatically for performance

### Bulk Insert Optimization

For large maps, use bulk insert for significant performance gains:

```python
# Standard insert (slower, better for debugging)
export_map_to_postgis(session, map_name, map_data, bulk_insert=False)

# Bulk insert (faster, production use)
export_map_to_postgis(session, map_name, map_data, bulk_insert=True)
```

## Spatial Queries

### Game Engine Integration

The query system is optimized for RPG game engines with sub-50ms response times:

```python
from py_fmg.db import create_query_helper

with db.get_session() as session:
    queries = create_query_helper(session)
    
    # Get all territorial info at player location
    territories = queries.get_territories_at_point(
        map_id=map_id,
        longitude=-122.4194,  # Player's longitude
        latitude=37.7749      # Player's latitude
    )
    
    # Returns: culture, religion, state, biome data
    print(f"Player is in {territories['culture']['name']}")
    print(f"Religion: {territories['religion']['name']}")
    print(f"Ruled by: {territories['state']['name']}")
    print(f"Biome: {territories['biome']['type']}")
```

### Settlement Queries

```python
# Find cities in a region
settlements = queries.find_settlements_in_region(
    map_id=map_id,
    bbox=(-123.0, 37.0, -122.0, 38.0),  # San Francisco Bay Area
    settlement_types=['capital', 'city'],
    min_population=1000,
    limit=50
)

for settlement in settlements:
    print(f"{settlement['name']}: {settlement['population']} people")
```

### River Networks

```python
# Get major rivers in region
rivers = queries.get_river_network(
    map_id=map_id,
    bbox=(-123.0, 37.0, -122.0, 38.0),
    min_discharge=100.0  # Minimum m³/s
)

for river in rivers:
    print(f"{river['name']}: {river['discharge_m3s']} m³/s")
```

### Climate Data

```python
# Get climate at location with regional averaging
climate = queries.get_climate_data(
    map_id=map_id,
    longitude=-122.4194,
    latitude=37.7749,
    radius_km=10.0  # Average within 10km
)

print(f"Temperature: {climate['temperature_c']}°C")
print(f"Precipitation: {climate['precipitation_mm']}mm/year")
```

## GeoPandas Export

Export spatial data for analysis in QGIS, Python, or other GIS tools:

```python
# Export cultures as GeoPandas DataFrame
cultures_gdf = queries.export_to_geopandas(
    map_id=map_id,
    table_name='cultures',
    bbox=None  # Optional bounding box filter
)

# Save to various formats
cultures_gdf.to_file('cultures.geojson', driver='GeoJSON')
cultures_gdf.to_file('cultures.shp')  # Shapefile
cultures_gdf.to_parquet('cultures.parquet')  # Parquet
```

## Performance Optimization

### Spatial Indexes

The system automatically creates GIST spatial indexes:

```sql
-- Automatically created indexes
CREATE INDEX idx_cultures_geom ON cultures USING GIST (geometry);
CREATE INDEX idx_religions_geom ON religions USING GIST (geometry);
CREATE INDEX idx_states_geom ON states USING GIST (geometry);
CREATE INDEX idx_settlements_geom ON settlements USING GIST (geometry);
CREATE INDEX idx_rivers_geom ON rivers USING GIST (geometry);
CREATE INDEX idx_biomes_geom ON biomes USING GIST (geometry);
```

### Query Performance

Typical query performance with spatial indexes:
- **Point-in-polygon queries**: < 10ms
- **Bounding box queries**: < 50ms  
- **Complex spatial joins**: < 200ms

### Connection Pooling

For production use, configure connection pooling:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30
)
```

## QGIS Integration

Connect QGIS directly to your PostGIS database:

1. **Add PostGIS Connection**:
   - Layer → Add Layer → Add PostGIS Layers
   - Create new connection with your database credentials

2. **Load Spatial Tables**:
   - Select tables: cultures, religions, states, settlements, rivers, biomes
   - Choose appropriate geometry columns

3. **Styling**:
   - Use the `color` field for automatic styling
   - Create choropleth maps using population, area, etc.

4. **Analysis**:
   - Perform spatial analysis using PostGIS functions
   - Create custom queries in DB Manager

## Advanced Usage

### Custom Spatial Queries

```python
# Raw SQL for complex analysis
with db.get_session() as session:
    result = session.execute(text("""
        SELECT c.name as culture_name, 
               ST_Area(c.geometry) as area_km2,
               COUNT(s.id) as settlement_count
        FROM cultures c
        LEFT JOIN settlements s ON ST_Contains(c.geometry, s.geometry)
        WHERE c.map_id = :map_id
        GROUP BY c.id, c.name, c.geometry
        ORDER BY area_km2 DESC
    """), {'map_id': str(map_id)})
    
    for row in result:
        print(f"{row.culture_name}: {row.area_km2:.1f} km², {row.settlement_count} settlements")
```

### Multi-Map Analysis

```python
# Compare multiple maps
maps = queries.list_maps(limit=10)
for map_record in maps:
    stats = queries.get_map_statistics(map_record.id)
    print(f"{map_record.name}: {stats['total_population']} total population")
```

## Troubleshooting

### Common Issues

1. **PostGIS Extension Missing**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```

2. **Geometry Creation Errors**:
   - Check that cell coordinates are valid
   - Ensure SRID is set to 4326 (WGS84)

3. **Performance Issues**:
   - Verify spatial indexes exist
   - Use EXPLAIN ANALYZE on slow queries
   - Consider partitioning for very large datasets

### Database Maintenance

```sql
-- Update statistics for query planner
ANALYZE;

-- Rebuild spatial indexes if needed
REINDEX INDEX idx_cultures_geom;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes 
WHERE indexname LIKE 'idx_%_geom';
```

## Future Enhancements

The current implementation provides Level 1 (macro-scale) tables. Future expansions will include:

- **Level 2 Tables**: Districts, roads, buildings for street-level detail
- **Temporal Data**: Historical changes and evolution tracking  
- **3D Support**: Elevation models and underground features
- **Real-time Updates**: Live map modifications and synchronization

## API Integration

The PostGIS integration works seamlessly with the FastAPI service:

```python
# In your FastAPI endpoints
from py_fmg.db import create_query_helper

@app.get("/api/territories/{longitude}/{latitude}")
async def get_territories(longitude: float, latitude: float, map_id: str):
    with db.get_session() as session:
        queries = create_query_helper(session)
        return queries.get_territories_at_point(
            map_id=UUID(map_id),
            longitude=longitude,
            latitude=latitude
        )
```

This enables real-time spatial queries for web applications, game engines, and mobile apps with excellent performance characteristics.