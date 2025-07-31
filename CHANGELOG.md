# Changelog

All notable changes to py-fmg will be documented in this file.

## [Unreleased]

### Added
- Complete cell packing (reGraph) implementation matching FMG's algorithm
  - Cell type classification (inland, coast, deep ocean)
  - Deep ocean filtering for performance
  - Coastal point enhancement with intermediate points
  - Grid index mapping for data transfer between packed/unpacked grids
- Comprehensive test suite for cell packing functionality
- API documentation in `docs/API.md`
- Lloyd's relaxation algorithm for improved Voronoi point distribution
  - ~42% improvement in point distribution uniformity
  - Configurable iteration count (default: 3)
- Height pre-allocation during Voronoi generation
  - Heights array created immediately after grid generation
  - Enables stateful operations matching FMG
- Grid reuse logic for interactive workflows
  - `should_regenerate()` method on VoronoiGraph
  - `generate_or_reuse_grid()` function
  - Supports "keep land, reroll mountains" workflow
- End-to-end test suite with seed-based reproducibility
- Main pipeline API endpoint for complete map generation

### Changed
- VoronoiGraph converted from immutable NamedTuple to mutable dataclass
  - Enables in-place modifications during generation
  - Stores generation parameters (graph_width, graph_height, seed)
- Updated all tests to work with new mutable structure
- Enhanced documentation in THEBIGREPORT.md and VORONOI.md

### Fixed
- Height array initialization matches FMG's pre-allocation behavior
- Grid generation now properly tracks state for reuse detection
- Cell connectivity building handles border cells correctly
- Heightmap PRNG desynchronization resolved through fixed operation ordering
- FMG blob spreading algorithm now accurately reproduces original behavior
  - Proper handling of spreading patterns and edge cases
  - Correct implementation of FMG's quirks in blob expansion

## [0.1.0] - Initial Release

### Added
- Initial Python port of FMG core functionality
- Voronoi graph generation
- Heightmap generation with multiple terrain algorithms
- Template support for named map styles
- PostGIS backend integration
- FastAPI REST interface
- Basic test coverage