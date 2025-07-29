# Python Fantasy Map Generator (py-fmg)

Python port of the Fantasy Map Generator (FMG) with PostGIS backend and FastAPI REST interface.

## Features

- **Complete Voronoi Generation**: Full FMG-compatible Voronoi graph generation with Lloyd's relaxation
- **Height Pre-allocation**: Heights array initialized during grid creation for stateful operations
- **Grid Reuse**: Support for "keep land, reroll mountains" workflow matching FMG's interactive design
- **Cell Packing (reGraph)**: Performance optimization that reduces ~10,000 cells to ~4,500 by filtering deep ocean
- **Coastal Enhancement**: Automatic addition of intermediate points along coastlines
- **Heightmap Generation**: Full suite of terrain generation algorithms (hills, pits, ranges, straits, etc.)
- **Template Support**: Named templates for quick map generation

## Setup

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

## Development

- Run tests: `poetry run pytest`
- Format code: `poetry run black . && poetry run isort .`
- Type check: `poetry run mypy py_fmg`
- Lint: `poetry run ruff check py_fmg`