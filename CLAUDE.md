# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Start the FastAPI server
poetry run uvicorn py_fmg.api.main:app --reload

# Run with specific host/port
poetry run uvicorn py_fmg.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_simple_hill.py

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=py_fmg
```

### Code Quality
```bash
# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy py_fmg

# Linting
poetry run ruff check py_fmg

# Security checks
poetry run bandit -r py_fmg
poetry run pip-audit
```

### Development
```bash
# Install dependencies
poetry install

# Update dependencies
poetry update

# Add new dependency
poetry add <package>

# Add dev dependency
poetry add --group dev <package>
```

## Architecture

### Project Structure
```
py_fmg/
├── api/           # FastAPI endpoints and request handlers
├── config/        # Configuration files (heightmap templates, etc.)
├── core/          # Core algorithms ported from FMG
│   ├── alea_prng.py         # Alea PRNG for FMG compatibility
│   ├── cell_packing.py      # reGraph algorithm for performance
│   ├── climate.py           # Temperature and precipitation calculations
│   ├── features.py          # Geographic feature detection
│   ├── heightmap_generator.py # Terrain generation algorithms
│   └── voronoi_graph.py     # Voronoi diagram generation
├── db/            # Database models and connections
└── utils/         # Utility functions
```

### Core Algorithm Flow
1. **Voronoi Generation** (`voronoi_graph.py`)
   - Jittered grid with boundary points
   - Delaunay triangulation via scipy
   - Lloyd's relaxation (3 iterations)
   - Height pre-allocation during generation

2. **Heightmap Generation** (`heightmap_generator.py`)
   - Template-based terrain commands (Hill, Range, Pit, etc.)
   - BFS blob spreading for FMG compatibility
   - Integer truncation matching Uint8Array behavior
   - Fixed PRNG consumption order

3. **Cell Packing** (`cell_packing.py`)
   - reGraph optimization reducing ~10k to ~4.5k cells
   - Deep ocean filtering
   - Coastal enhancement with intermediate points

4. **Feature Detection** (`features.py`)
   - Ocean/land classification (height < 20)
   - Lake and coastline detection
   - Island identification
   - Distance field calculation for coastlines

5. **Climate System** (`climate.py`)
   - Latitude-based temperature bands with tropical/temperate/polar zones
   - Altitude temperature drop (6.5°C per 1km)
   - Wind patterns by latitude (trade winds, westerlies, polar easterlies)
   - Precipitation with orographic effects and rain shadows
   - Moisture transport simulation

### Key Implementation Details

#### PRNG Synchronization
The project uses a custom Alea PRNG implementation to match FMG's random number generation exactly. This ensures deterministic terrain generation with the same seeds.

#### FMG Compatibility Quirks
- `getLinePower()` bug replicated for range generation
- Integer truncation in blob spreading matches Uint8Array
- Pit function uses specific visited pattern
- Pipeline follows FMG order: markupGrid → reGraph → markupPack

#### Database Integration
- PostgreSQL with PostGIS for spatial data
- Prepared for future street-level detail
- UUID primary keys for all entities
- EPSG:4326 projection standard

### Testing Strategy
- Seed-based reproducibility tests
- Comparison with FMG reference outputs
- Unit tests for individual algorithms
- End-to-end map generation tests

### Performance Considerations
- BFS blob spreading kept for accuracy (not vectorized)
- Cell packing reduces computational load ~55%
- Target: < 60s world generation
- PostGIS indexes for < 50ms spatial queries