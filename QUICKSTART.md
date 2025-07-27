# Quick Start Guide

## Prerequisites

1. **Python 3.10+** 
2. **PostgreSQL 14+** with **PostGIS 3+** extension
3. **Poetry** for dependency management

## Setup

### 1. Install Dependencies
```bash
# Install Python dependencies
poetry install

# Activate virtual environment
source venv/bin/activate
```

### 2. Configure Environment
```bash
# Copy and edit environment variables
cp .env.example .env

# Edit .env with your database credentials:
DB_USER='your_user'
DB_HOST='localhost'  
DB_NAME='py-fmg'
DB_PASSWORD='your_password'
DB_PORT=5432
```

### 3. Setup Database
```sql
-- Create database and enable PostGIS
CREATE DATABASE "py-fmg";
\c py-fmg
CREATE EXTENSION postgis;
```

### 4. Run Tests
```bash
# Test core Voronoi implementation
python -m pytest tests/test_voronoi_graph.py -v

# All tests should pass (16/16)
```

### 5. Start API Server
```bash
# Run development server
uvicorn py_fmg.api.main:app --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## Basic Usage

### Generate a Map via API

```bash
# Start map generation
curl -X POST "http://localhost:8000/maps/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "seed": "test123",
    "width": 800,
    "height": 600,
    "cells_desired": 10000,
    "map_name": "Test Map"
  }'

# Response: {"job_id": "uuid", "status": "pending", ...}

# Check job status  
curl "http://localhost:8000/jobs/{job_id}"

# List generated maps
curl "http://localhost:8000/maps"
```

### Use Core Components Directly

```python
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph

# Generate Voronoi graph
config = GridConfig(width=800, height=600, cells_desired=1000)
graph = generate_voronoi_graph(config, seed="test123")

print(f"Generated {len(graph.points)} cells")
print(f"Border cells: {sum(graph.cell_border_flags)}")

# Access cell connectivity
for i, neighbors in enumerate(graph.cell_neighbors[:5]):
    print(f"Cell {i} neighbors: {neighbors}")
```

## Current Capabilities

### ✅ Working Features
- **Voronoi Graph Generation**: Complete grid generation with proper cell connectivity
- **Database Models**: PostGIS-enabled schema for map data storage  
- **REST API**: Async map generation with job tracking
- **Configuration**: Environment-based settings management

### 🚧 In Development
- **Heightmap Generation**: Template-based terrain creation
- **Climate Simulation**: Temperature and precipitation models
- **Hydrology**: River generation and water flow simulation
- **Political Systems**: Settlement placement and state boundaries

## Development Workflow

### Running Tests
```bash
# Run specific test file
python -m pytest tests/test_voronoi_graph.py -v

# Run all tests
python -m pytest -v

# Run with coverage
python -m pytest --cov=py_fmg tests/
```

### Code Quality
```bash
# Format code
black py_fmg tests

# Sort imports  
isort py_fmg tests

# Type checking
mypy py_fmg

# Linting
ruff check py_fmg
```

### Database Operations
```bash
# Reset database (drops all tables)
python -c "
from py_fmg.db.connection import db
from py_fmg.db.models import Base
db.initialize()
Base.metadata.drop_all(bind=db.engine)
Base.metadata.create_all(bind=db.engine)
"
```

## Project Structure

```
py-fmg/
├── py_fmg/                 # Main Python package
│   ├── core/              # Core generation algorithms
│   │   ├── voronoi_graph.py      # ✅ Voronoi system
│   │   ├── heightmap_analysis.py # 📋 Algorithm docs
│   │   └── ...
│   ├── db/                # Database models and connections
│   ├── api/              # FastAPI web service
│   └── config.py         # Configuration management
├── tests/                # Test suite
├── fmg/                  # Original FMG JavaScript source
├── pyproject.toml        # Poetry dependencies
└── .env                  # Environment configuration
```

## Next Steps

1. **Implement Heightmap Generation** - Port FMG's template system
2. **Add Climate Simulation** - Temperature/precipitation models  
3. **Build Hydrology System** - Rivers and water flow (most complex)
4. **Create Settlement System** - Cities and political boundaries
5. **Add Testing Framework** - Compare outputs with original FMG

See `IMPLEMENTATION_STATUS.md` for detailed roadmap and `TASKS.md` for complete task breakdown.

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running
- Verify PostGIS extension is installed
- Check .env file database credentials

### Import Errors
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `poetry install`
- Check Python path includes project root

### Test Failures
- Ensure scipy and numpy are installed correctly
- Check random seed consistency in tests
- Verify Voronoi diagram generation works