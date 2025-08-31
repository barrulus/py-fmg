# Python Fantasy Map Generator (py-fmg)

A **production-ready** procedural fantasy world generation service created by porting the world-generation algorithms from [Azgaar's Fantasy Map Generator (FMG)](https://github.com/Azgaar/Fantasy-Map-Generator) into a standalone Python application. This service generates complete fantasy worlds with realistic geography, climate, cultures, and settlements, storing all data in a PostGIS-enabled PostgreSQL database for advanced spatial queries and RPG integration.

## 🎯 Project Status

**v1.0 - Production Ready** - Complete FMG algorithm port with enhanced features:
- ✅ Full Voronoi-based terrain generation with 6000+ cells
- ✅ Realistic climate simulation with wind patterns and precipitation
- ✅ Biome classification with habitability scoring
- ✅ Hydrological systems with river networks
- ✅ Cultural and political boundaries with dynamic state generation
- ✅ Settlement hierarchies with population distribution
- ✅ Comprehensive PostGIS database export
- ✅ Docker/Kubernetes deployment support

## 🚀 Features

### Core Generation Algorithms
- **Voronoi Graph Generation**: FMG-compatible jittered grid with Lloyd's relaxation
- **Heightmap Generation**: Multiple terrain algorithms (hills, ranges, pits, straits)
- **Cell Packing (reGraph)**: Performance optimization reducing 10k → 4.5k cells
- **Climate System**: Temperature bands, wind patterns, orographic precipitation
- **Biome Classification**: 12 biome types with realistic distribution
- **Hydrology**: River networks with proper flow to lakes/ocean
- **Cultures & States**: Political territories with population-based borders
- **Settlements**: Hierarchical placement (capitals → towns → villages)
- **Name Generation**: Markov chain-based fantasy name generator

### Technical Features
- **PRNG Synchronization**: Alea PRNG for FMG-compatible deterministic generation
- **PostGIS Integration**: Full spatial database with 7 data types
- **RESTful API**: FastAPI with comprehensive endpoints
- **Visualization**: Map rendering with political borders, rivers, settlements
- **Docker Support**: Production-ready containerization
- **Extensive Testing**: Seed-based reproducibility tests

## 🏗️ Architecture

```
┌─────────────────┐      ┌─────────────────────────────┐      ┌────────────────────┐
│   FastAPI       ├──────►   Generation Pipeline       ├──────►   PostGIS Database │
│   Endpoints     │      │   • Voronoi → Heightmap     │      │   • Cells          │
│                 │      │   • Climate → Biomes        │      │   • Climate        │
│                 │      │   • Rivers → Cultures       │      │   • Rivers         │
│                 │      │   • Settlements → States    │      │   • Cultures       │
└─────────────────┘      └─────────────────────────────┘      │   • Biomes         │
                                                              │   • Religions      │
                                                              │   • Settlements    │
                                                              └────────────────────┘
```

## 📦 Installation

### Local Development

1. **Prerequisites**
   - Python 3.10+
   - PostgreSQL 14+ with PostGIS 3+
   - Poetry for dependency management

2. **Setup**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/py-fmg.git
   cd py-fmg

   # Install dependencies
   poetry install

   # Configure environment
   cp .env.example .env
   # Edit .env with your database credentials

   # Initialize database
   poetry run python -m py_fmg.db.init_db
   ```

3. **Run Development Server**
   ```bash
   poetry run uvicorn py_fmg.api.main:app --reload
   ```

### Docker Deployment

```bash
# Development
docker-compose up

# Production
docker-compose -f docker-compose.prod.yaml up
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

## 🌐 API Usage

### Generate a World

```bash
# Generate with default settings
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "width": 1200,
    "height": 1000,
    "seed": 123456789,
    "template": "continents"
  }'

# Response includes world_id for database queries
```

### Available Templates
- `archipelago` - Island chains
- `atoll` - Ring-shaped islands
- `continents` - Multiple continents (default)
- `peninsula` - Coastal landmasses
- `pangea` - Single supercontinent

### Query Generated Data

```bash
# Get world details
GET /api/worlds/{world_id}

# Get specific data types
GET /api/worlds/{world_id}/cells
GET /api/worlds/{world_id}/rivers
GET /api/worlds/{world_id}/settlements
GET /api/worlds/{world_id}/cultures
```

## 🗺️ Spatial Queries

The PostGIS database enables powerful RPG integration:

```sql
-- Find player's current state
SELECT s.* FROM states s 
WHERE ST_Contains(s.geometry, ST_MakePoint(lon, lat));

-- Get nearby settlements
SELECT name, type, population, ST_Distance(geometry, player_pos) as distance
FROM settlements
WHERE ST_DWithin(geometry, player_pos, 10000)
ORDER BY distance;

-- Find rivers within view distance
SELECT r.* FROM rivers r
WHERE ST_DWithin(r.geometry, player_pos, 5000);

-- Get biome at location
SELECT b.name, b.habitability 
FROM biomes b
WHERE ST_Contains(b.geometry, player_pos);
```

## 🧪 Development

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test suite
poetry run pytest tests/test_api_integration.py -v

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
```

## 📊 Performance

- **Generation Time**: < 60 seconds for full world
- **Map Size**: 1200×1000 generates ~6000 Voronoi cells
- **Database Queries**: < 50ms with spatial indexes
- **Memory Usage**: ~2GB peak during generation
- **API Response**: Streaming for large datasets

## 🔮 Roadmap

### Phase 1 optional : clean code and better practice
- [ ] routeur as api gateway -> fast api
- [ ] model folder and data class files
- [ ] better management of each event and type (voronoi_graph)
- [ ] fallback when order is wrong in collapse function and try catch behaviour.

### Phase 2: City Generation (In Progress)
- [ ] Street-level detail generation
- [ ] Building footprints with types
- [ ] Urban district classification
- [ ] Points of interest placement

### Phase 3: Advanced Features
- [ ] Trade route generation
- [ ] Economic simulation
- [ ] Historical timeline generation
- [ ] Dungeon placement algorithms
- [ ] Weather patterns simulation

### Phase 4: RPG Integration
- [ ] Unity/Unreal Engine plugins
- [ ] Real-time detail streaming
- [ ] Procedural quest generation
- [ ] NPC population simulation

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [API Documentation](docs/API.md)
- [PostGIS Integration Guide](docs/POSTGIS_INTEGRATION.md)
- [Algorithm Documentation](docs/ALGORITHMS.md)
- [DevOps Guide](DEVOPS.md)

## 🙏 Acknowledgments

- [Azgaar's Fantasy Map Generator](https://github.com/Azgaar/Fantasy-Map-Generator) - Original algorithms and inspiration
- FMG community for extensive documentation
- Contributors who helped port and enhance the algorithms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **API Framework**: FastAPI
- **Database**: PostgreSQL 14+ with PostGIS 3+
- **Geospatial**: GeoPandas, Shapely, Rasterio
- **Scientific**: NumPy, SciPy, scikit-learn
- **Visualization**: Matplotlib, Pillow
- **Testing**: Pytest, pytest-cov
- **Deployment**: Docker, Kubernetes, Nginx
