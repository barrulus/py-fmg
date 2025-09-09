# Python Fantasy Map Generator (py-fmg)

Headless Python port of Azgaar's Fantasy Map Generator (FMG) focused on producing validated GeoJSON for ingestion into PostGIS. Includes Leaflet-based HTML previews and optional static PNGs for debugging and sharing.

Targets: cells, burgs, states, provinces, routes, rivers, regiments, markers.

## Project Vision

### Core Philosophy

- True logic port of FMG algorithms (no UI automation)
- Hierarchical procedural generation (continents → regions → cities)
- Data-first architecture: PostGIS as the source of truth
- GeoJSON interchange; no QGIS dependency (QGIS optional for analysis)
- Leaflet previews for quick iteration and validation

## Features

- Complete Voronoi grid generation with Lloyd relaxation
- Height pre-allocation and grid reuse (“keep land, reroll mountains”)
- Cell packing (reGraph) to drop deep ocean cells
- Coastal enhancement via intermediate points
- Heightmap templates (hills, pits, ranges, straits, etc.)
- Deterministic PRNG sequencing across steps
- Accurate FMG blob spreading behavior
- Validated GeoJSON export; Leaflet HTML preview generator
- Seeded tests and snapshot validation

## Quick Start

### Prerequisites

1) Python 3.13+
2) PostgreSQL 17+ with PostGIS 3+
3) Poetry (dependency management)

### Setup

1) Install dependencies

```bash
poetry install
```

2) Configure environment

```bash
cp .env.example .env
# update values
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=py-fmg
```

3) Initialize database (ensure PostGIS enabled)

```sql
CREATE DATABASE "py-fmg";
\c py-fmg
CREATE EXTENSION postgis;
```

4) Run tests (optional, recommended)

```bash
poetry run pytest -q
```

5) Start API server

```bash
poetry run uvicorn py_fmg.api.main:app --reload
# http://localhost:8000  (docs: /docs)
```

### Basic Usage

- API map generation

```bash
curl -X POST "http://localhost:8000/maps/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "seed": "test123",
    "width": 800,
    "height": 600,
    "cells_desired": 10000,
    "map_name": "Test Map",
    "template": "volcano"
  }'

# Check job status
curl "http://localhost:8000/jobs/{job_id}"

# List generated maps
curl "http://localhost:8000/maps"
```

- Core components in Python

```python
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph

config = GridConfig(width=800, height=600, cells_desired=1000)
graph = generate_voronoi_graph(config, seed="test123")
print(len(graph.points), sum(graph.cell_border_flags))
```

- CLI quick run (GeoJSON + Leaflet preview)

```bash
python -m cli.main \
  --width 1200 --height 800 --cells 20000 \
  --template continents --preview --geojson \
  --snap-to-coast-steps 0   # 0 = unlimited
```

## CLI Reference

Common switches (see full list via `python -m cli.main --help`):

```text
# Core map params
--width <float>               # Map width (px) (default: 1000)
--height <float>              # Map height (px) (default: 800)
--cells <int>                 # Target number of cells (default: 10000)
--seed <str>                  # Random seed (string)
--out <path>                  # Output root directory (default: out)
--template <str>              # Heightmap template (default: continents)
--target-land <0..1>          # Target land fraction (auto sea level)

# Preview
--preview [basename]          # Generate Leaflet layers preview
--preview-scale <float>       # Preview scale factor (default: 1.0)
--preview-layers <list>       # Layers to include (space- or comma-separated). Use 'all' or any subset:
                              # cells, topography, hillshade, provinces, climate, biomes, cultures_cells,
                              # watermask, burgs, rivers_smooth, rivers_polygons, routes, sea_routes, markers, regiments, coastlines
--no-relax                    # Disable Lloyd relaxation

# GeoJSON export
--geojson [basename]          # Write GeoJSON artifacts (optional basename)

# FMG .map export
--export-map [path]           # Export FMG .map (optional)
--export-map-minimal          # Minimal .map with safe defaults

# Hydrology
--min-river-flux <float>      # Minimum flux to form visible river
--precip-mult <float>         # Precipitation multiplier
--snap-to-coast-steps <int>   # Steps to extend mouths toward ocean (0 = unlimited)
--resolve-steps <int>         # Max iterations for depression resolution

# Climate tuning
--equator-temp <float>        # Sea-level temperature at equator (°C)
--tropical-gradient <float>   # Temperature drop per degree in tropics (°C/°)
--itcz-width <float>          # ITCZ half-width around equator (degrees)
--itcz-boost <float>          # ITCZ precipitation multiplier

# Settlements and states
--states-number <int>         # Target number of states (capitals)
--burgs-number <int>          # Target number of towns (1000 = auto)
--town-spacing-base <int>     # Base divisor (lower = more towns)
--town-spacing-power <float>  # Power adjustment (lower = more towns)
--urbanization-rate <float>   # Urbanization rate (0..1)
```

Tips
- More headwaters: lower `--min-river-flux` (e.g., 20) and raise `--precip-mult` (1.2–1.5)
- Wetter tropics: increase `--itcz-boost` (1.5–2.0) and widen `--itcz-width` (12–20)
- More towns: increase `--burgs-number` and lower `--town-spacing-base`

### Precreated Heightmaps

- List bundled heightmaps: `python -m cli.main --list-precreated`
- Use a precreated heightmap: `python -m cli.main --precreated europe --geojson --preview`

## NixOS Setup

1) Enter development shell

```bash
nix develop
```

2) Install dependencies (inside shell)

```bash
poetry install
```

3) Start Postgres with PostGIS (systemd or container)

```bash
# System service
sudo systemctl start postgresql

# Or Podman/Docker
podman run -d --name postgres-postgis \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=py_fmg \
  -p 5432:5432 \
  postgis/postgis:17-3.3
```

4) Run API

```bash
poetry run uvicorn py_fmg.api.main:app --reload
```

## RPG Integration

Example PostGIS queries from a game engine:

```sql
-- What state is the player in?
SELECT * FROM states WHERE ST_Contains(geom, player_location);

-- Any taverns within 50 meters?
SELECT * FROM buildings WHERE type = 'tavern' AND ST_DWithin(geom, player_location, 50);

-- Nearest road
SELECT * FROM roads ORDER BY geom <-> player_location LIMIT 1;
```
