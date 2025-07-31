# Python Fantasy Map Generator (py-fmg) - Current Status

## Project Overview

Python port of the Fantasy Map Generator (FMG) with PostGIS backend and FastAPI REST interface. The project aims to provide a complete, accurate implementation of FMG's terrain generation algorithms with modern Python architecture.

## Current Features

### Core Functionality
- **Complete Voronoi Generation**: Full FMG-compatible Voronoi graph generation with Lloyd's relaxation
  - ~42% improvement in point distribution uniformity
  - Configurable iteration count (default: 3)
  - Heights array pre-allocated during grid creation for stateful operations

- **Grid Management**
  - Support for "keep land, reroll mountains" workflow matching FMG's interactive design
  - `should_regenerate()` method on VoronoiGraph for intelligent reuse
  - `generate_or_reuse_grid()` function for workflow optimization

- **Cell Packing (reGraph)**: Performance optimization that reduces ~10,000 cells to ~4,500
  - Cell type classification (inland, coast, deep ocean)
  - Deep ocean filtering for performance
  - Grid index mapping for data transfer between packed/unpacked grids

- **Coastal Enhancement**: Automatic addition of intermediate points along coastlines

- **Heightmap Generation**: Full suite of terrain generation algorithms
  - Hills, pits, ranges, straits implementation
  - PRNG synchronization through fixed-order operations
  - FMG blob spreading with accurate quirks handling

- **Template Support**: Named templates for quick map generation

- **API Interface**: FastAPI REST endpoints for map generation
  - Main pipeline endpoint for complete generation
  - Support for template-based generation

### Infrastructure
- **Database**: PostGIS backend integration for spatial operations
- **Testing**: Comprehensive test suite with seed-based reproducibility
- **Documentation**: API documentation and detailed technical reports

## Recent Accomplishments

### Bug Fixes
- **PRNG Desynchronization**: Fixed heightmap generation order to ensure deterministic results
- **Blob Spreading**: Implemented FMG's exact blob spreading algorithm including its quirks
- **Cell Connectivity**: Proper handling of border cells in graph building

### Testing
- Added end-to-end test suite with seed validation
- Main pipeline API testing with deterministic verification
- Comprehensive cell packing test coverage

### Architecture Improvements
- Converted VoronoiGraph from immutable NamedTuple to mutable dataclass
- Enabled in-place modifications during generation
- Proper state tracking for grid reuse detection

## Technical Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Database**: PostgreSQL with PostGIS
- **Package Management**: Poetry
- **Testing**: pytest
- **Code Quality**: black, isort, mypy, ruff

## Setup Instructions

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

## Development Commands

- **Run tests**: `poetry run pytest`
- **Format code**: `poetry run black . && poetry run isort .`
- **Type check**: `poetry run mypy py_fmg`
- **Lint**: `poetry run ruff check py_fmg`

## Project Status

The project has reached a significant milestone with accurate FMG algorithm implementation. The core terrain generation features are complete and tested. The system successfully generates maps that match FMG's output for given seeds, demonstrating algorithmic accuracy.

### Completed
- ✅ Voronoi graph generation with Lloyd's relaxation
- ✅ Cell packing (reGraph) implementation
- ✅ Heightmap generation algorithms
- ✅ PRNG synchronization
- ✅ FMG blob spreading with quirks
- ✅ Grid reuse workflow
- ✅ API endpoints
- ✅ Comprehensive testing

### Future Considerations
- Additional terrain features (rivers, biomes, etc.)
- Performance optimizations
- Extended API functionality
- UI/visualization layer

## Key Files and Documentation

- `README.md` - Basic project overview and setup
- `CHANGELOG.md` - Detailed change history
- `docs/API.md` - API endpoint documentation
- `docs/THEBIGREPORT.md` - Technical implementation details
- `docs/VORONOI.md` - Voronoi generation documentation