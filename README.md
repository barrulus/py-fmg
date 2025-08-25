# Python Fantasy Map Generator (py-fmg)

A headless procedural map generation service created by porting the world-generation algorithms from [Azgaar's Fantasy Map Generator (FMG)](https://github.com/Azgaar/Fantasy-Map-Generator) into a standalone Python application. This service leverages Python's premier geospatial libraries to store generated map data in a PostGIS-enabled PostgreSQL database, designed specifically for use by procedural Role-Playing Games (RPGs).

## Project Vision

### Core Philosophy

- **True Logic Port**: Re-implementation of FMG's core algorithms in Python, not UI automation
- **Hierarchical Procedural Generation**: Multi-level detail from continents down to future street-level expansion
- **Data-First Architecture**: PostGIS database as the canonical source of truth, not static files
- **PostGIS + QGIS Integration**: High-performance spatial database with QGIS for visualization and authoring

### Architecture Overview

```
┌─────────────────┐      ┌─────────────────────────────────┐      ┌────────────────────┐
│   API / CLI     ├──────►   Python Generation Service     ├──────►   PostGIS Database │
│ (FastAPI/Click) │      │ (FMG Logic + City Generators)   │      │ (PostgreSQL)       │
└─────────────────┘      └─────────────────────────────────┘      └────────────────────┘
```

## Features

- **Complete Voronoi Generation**: Full FMG-compatible Voronoi graph generation with Lloyd's relaxation
- **Height Pre-allocation**: Heights array initialized during grid creation for stateful operations
- **Grid Reuse**: Support for "keep land, reroll mountains" workflow matching FMG's interactive design
- **Cell Packing (reGraph)**: Performance optimization that reduces ~10,000 cells to ~4,500 by filtering deep ocean
- **Coastal Enhancement**: Automatic addition of intermediate points along coastlines
- **Heightmap Generation**: Full suite of terrain generation algorithms (hills, pits, ranges, straits, etc.)
- **Template Support**: Named templates for quick map generation
- **PRNG Synchronization**: Fixed-order heightmap operations ensure deterministic terrain generation
- **FMG Blob Spreading**: Accurate implementation of FMG's blob spreading algorithm with proper quirks handling
- **Comprehensive Testing**: End-to-end tests with seed-based reproducibility

## Standard Setup

1. Install dependencies:

```bash
poetry install
```

2. Configure environment:
   Copy `.env.example` to `.env` and update database credentials.

3. Start PostgreSQL with PostGIS extension.

4. Run the API:

```bash
poetry run uvicorn py_fmg.api.main:app --reload
```

## NixOs Setup

### NixOS Setup

1. Enable the Nix development shell:

```bash
nix develop
```

2. Install Poetry dependencies within the Nix shell:

```bash
poetry install
```

3. Configure PostgreSQL service in your NixOS configuration or use a containerized approach:

```bash
# Option 1: System PostgreSQL (requires NixOS config)
sudo systemctl start postgresql
# Option 2: Docker/Podman container
podman run -d --name postgres-postgis \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=py_fmg \
  -p 5432:5432 \
  postgis/postgis:15-3.3
```

4. Run the API:

```bash
poetry run uvicorn py_fmg.api.main:app --reload
```

## Development

- Run tests: `poetry run pytest`
- Format code: `poetry run black . && poetry run isort .`
- Type check: `poetry run mypy py_fmg`
- Lint: `poetry run ruff check py_fmg`

## RPG Integration

The system is designed for real-time spatial queries from game engines:

```sql
-- What state is the player in?
SELECT * FROM states WHERE ST_Contains(geom, player_location);

-- Are there any taverns within 50 meters?
SELECT * FROM buildings WHERE type = 'tavern' AND ST_DWithin(geom, player_location, 50);

-- Find the nearest road
SELECT * FROM roads ORDER BY geom <-> player_location LIMIT 1;
```

## Future Roadmap

### Phase 2: Street-Scale Generation

- Specialized city generator module with urban algorithms
- Agent-based systems for organic road networks
- Building footprint generation
- On-demand detail generation via API

### Phase 3: Extended Features

- Rivers and water bodies
- Advanced biome generation
- Cultural and political boundaries
- Trade routes and economic simulation

## Technology Stack

- **Language**: Python 3.10+
- **API Framework**: FastAPI
- **Geospatial Processing**: GeoPandas, Shapely, Rasterio
- **Database**: PostgreSQL 14+ with PostGIS 3+
- **Numerical/Scientific**: NumPy, SciPy
- **Visualization**: QGIS 3.x
