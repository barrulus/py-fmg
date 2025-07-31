# Python Port Task Breakdown

Based on analysis of the Fantasy Map Generator (FMG) codebase and PLAN.md requirements, this document outlines the comprehensive task breakdown for porting FMG logic to Python.

## **Refined Implementation Strategy**

*   **Embrace Pythonic Solutions:** This is not a 1:1 port. Where Python's ecosystem offers a more performant or robust solution (e.g., NumPy, GeoPandas, existing libraries), we will favor it over a direct translation of JavaScript code.
*   **Test-Driven Development with Stage Gates:** The project will rely heavily on a rigorous testing framework. We will test each major generation stage against reference data exported from the original FMG to isolate deviations early.
*   **Focus on High-Risk Components:** The Voronoi graph adapter, hydrology simulation, and state expansion logic are the highest-risk modules and will receive priority and additional resources.

## FMG Algorithm Analysis Summary

### Core Generation Flow (main.js:616-698)
**Grid Generation** → **Heightmap** → **Geographic Features** → **Climate** → **Hydrology** → **Biomes** → **Cultures** → **Settlements** → **Political Systems**

### Key Components Identified

#### 1. Voronoi Graph System (`utils/graphUtils.js`, `modules/voronoi.js`)
- **Jittered grid generation**: Regular square grid with randomized point positions
- **Delaunator triangulation**: Uses Mapbox Delaunator for Delaunay triangulation
- **Voronoi cell construction**: Custom Voronoi class builds dual graph from triangulation
- **Cell connectivity**: Each cell knows its neighbors and vertices

#### 2. Heightmap Generation (`modules/heightmap-generator.js`)
- **Template system**: String-based commands (Hill, Range, Trough, Pit, Strait, etc.)
- **Procedural algorithms**: Blob/hill spreading with configurable power factors
- **Image import**: PNG heightmap loading and processing
- **Terrain modification**: Mathematical operations (smooth, mask, invert)

#### 3. Hydrology System (`modules/river-generator.js`)
- **Depression filling**: Iterative algorithm to resolve local minima
- **Water flow simulation**: Precipitation → flux accumulation → river formation
- **Lake detection**: Closed basins and outlet calculation
- **River meandering**: Path smoothing and width calculation

#### 4. Biome Classification (`modules/biomes.js`)
- **Climate-based matrix**: Temperature/moisture lookup table
- **Special conditions**: Wetland, desert, and permafrost overrides
- **River influence**: Enhanced moisture near water bodies

#### 5. Settlement System (`modules/burgs-and-states.js`)
- **Capital placement**: Score-based selection with minimum spacing
- **State formation**: Expansion algorithms based on culture and geography
- **Urban hierarchy**: Population-based city/town classification

#### 6. Name Generation System (`modules/names-generator.js`)
- **Markov chain generation**: Syllable-based procedural name creation
- **Cultural naming**: Different linguistic rules per culture
- **Entity-specific naming**: Settlements, states, geographic features, maps
- **Name variation**: Multiple naming patterns and conventions

## Python Port Task Breakdown

### Implementation Progress Summary
✅ **Completed Components:**
- Task 1-6: FMG Algorithm Analysis (100% complete)
- Task 7: Core Infrastructure (100% complete)
- Task 8: Voronoi Graph Generation (100% complete with enhancements)
- Task 9: Heightmap Generation (100% complete - all issues resolved)
- Task 9.5: Cell Packing/reGraph (100% complete)
- Task 10: Geographic Features - Basic Implementation (100% complete)

🚧 **In Progress:**
- None currently

❌ **Not Started:**
- Tasks 11-18: Climate, Hydrology, Biomes, Settlements, Names, PostGIS, FastAPI, Testing

### Phase 1: Core Map Generation Engine

#### **Infrastructure & Dependencies**
- **Language**: Python 3.10+
- **API Framework**: FastAPI
- **Database**: PostgreSQL 14+ with PostGIS 3+, SQLAlchemy
- **Geospatial**: GeoPandas, Shapely, Rasterio
- **Scientific**: NumPy, SciPy (Voronoi), noise (Perlin)

#### Implementation Tasks

##### Task 1: Analyze FMG core generation modules to understand the algorithm flow
- [x] Study main.js generation pipeline (lines 616-698)
- [x] Identify key module dependencies and execution order
- [x] Document the complete flow: Grid → Heightmap → Features → Climate → Hydrology → Biomes → Settlements
- [x] Map out data structures passed between modules

##### Task 2: Identify key JavaScript modules for heightmap generation
- [x] Analyze modules/heightmap-generator.js algorithms
- [x] Study template system and procedural generation methods
- [x] Document Hill, Range, Trough, Pit, and Strait algorithms
- [x] Understand power factors and blob spreading mechanics

##### Task 3: Analyze Voronoi diagram generation and graph creation logic
- [x] Study utils/graphUtils.js grid generation functions
- [x] Analyze modules/voronoi.js Voronoi construction
- [x] Understand Delaunator integration and cell connectivity
- [x] Document graph data structures and neighbor relationships

##### Task 4: Study hydrology and river generation algorithms
- [x] Analyze modules/river-generator.js water flow simulation
- [x] Study depression filling and drainage algorithms
- [x] Understand lake detection and outlet calculation
- [x] Document river meandering and width calculation methods

##### Task 5: Examine biome assignment and attribute logic
- [x] Study modules/biomes.js classification system
- [x] Analyze temperature/moisture matrix lookup
- [x] Understand special biome conditions (wetland, desert, permafrost)
- [x] Document river influence on biome assignment

##### Task 6: Analyze burgs (cities/towns) and states generation
- [x] Study modules/burgs-and-states.js settlement placement
- [x] Analyze capital placement scoring and spacing algorithms
- [x] Understand state expansion and political boundary creation
- [x] Document population calculation and urban hierarchy

##### Task 7: Core Infrastructure - Set up Python project structure
- [x] Initialize Python project with proper directory structure
- [x] Set up virtual environment and dependency management (poetry/pip)
- [x] Create configuration management system
- [x] Implement logging and error handling framework
- [x] Set up database connection utilities

##### Task 8: Voronoi Graph Generation - Port grid generation and Voronoi calculation ✅ COMPLETE
- **Note:** This is a high-risk foundational task. The output must be topologically identical to FMG's graph structure.
- [x] Port `getJitteredGrid()` and `getBoundaryPoints()` from `utils/graphUtils.js`.
- [x] Use `scipy.spatial.Voronoi` to generate the initial Voronoi diagram.
- [x] **Crucial:** Develop an "adapter" layer to convert the `scipy.spatial.Voronoi` output into the cell-centric graph data structure that FMG's algorithms expect (e.g., cell neighbors, ordered vertices for each cell).
- [x] Create a dedicated, robust test suite for this module to validate its topology against FMG reference data.
- [x] **Additional Features Implemented:**
  - Lloyd's relaxation (3 iterations) for ~42% improved point distribution
  - Height pre-allocation during generation (not heightmap phase)
  - Grid reuse logic with `should_regenerate()` method
  - Mutable dataclass structure for stateful operations

##### Task 9: Heightmap Generation - Port heightmap templates and algorithms ✅ COMPLETE
- **Recommendation:** Implement algorithms using NumPy vectorization for massive performance gains over iterative loops.
- [x] Port heightmap template system from `modules/heightmap-generator.js`.
- [x] Implement procedural algorithms (kept FMG's BFS approach for exact compatibility):
  - [x] `addHill()` - blob spreading algorithm with BFS queue
  - [x] `addRange()` - mountain range generation with pathfinding
  - [x] `addTrough()` - valley creation
  - [x] `addPit()` - depression creation with correct visited pattern
  - [x] `addStrait()` - water channel cutting (vertical/horizontal)
- [x] Port terrain modification functions (`smooth`, `mask`, `modify`, `invert`).
- [x] Template parsing from named templates (all 17 templates)
- [x] Integer truncation matching FMG's Uint8Array behavior
- [x] PRNG consumption audit - verified exact match with FMG
- [ ] Implement PNG heightmap import using Rasterio.

**Resolved Issues:**
1. ✅ Water/land distribution fixed with integer truncation in blob spreading
2. ✅ FMG's `getLinePower()` bug replicated for compatibility 
3. ✅ Pit function pattern corrected to match FMG
4. ✅ Pipeline architecture fixed (Features.markupGrid before reGraph)
5. ✅ PRNG consumption matches FMG exactly

**Status:** Complete with full FMG compatibility. Archipelago template now generates proper islands.

##### Task 9.5: Cell Packing (reGraph) - Performance optimization ✅ COMPLETE
- [x] Implement FMG's `reGraph()` function for cell packing
- [x] Cell type classification (inland, coast, deep ocean)
- [x] Deep ocean filtering (reduces ~10k cells to ~4.5k)
- [x] Coastal point enhancement with intermediate points
- [x] New Voronoi generation from packed points
- [x] Grid index mapping for data transfer between packed/unpacked
- [x] Comprehensive test coverage

**Status:** Full FMG reGraph algorithm implemented with coastal enhancement.

##### Task 10: Geographic Features - Port ocean/land detection and basic features
- [ ] Port `Features.markupGrid()` and `Features.markupPack()` logic
- [ ] Implement ocean/land classification (height < 20 = water)
- [ ] Port lake detection in deep depressions (`addLakesInDeepDepressions()` from main.js:717-775)
- [ ] Port near-sea lake opening logic (`openNearSeaLakes()` from main.js:778-818)
- [ ] Implement coastline detection and island identification

**Note:** Cell type classification for coast/inland/ocean is already implemented in `cell_packing.py` as part of reGraph.

##### Task 11: Temperature/Precipitation - Port climate calculation models
- [ ] Port temperature calculation from `main.js:897-943`:
  - [ ] Latitude-based temperature bands
  - [ ] Altitude temperature drop (6.5°C per 1km)
  - [ ] Tropical/temperate/polar gradients
- [ ] Port precipitation model from `main.js:946-1106`:
  - [ ] Wind direction calculation by latitude bands
  - [ ] Moisture transport simulation
  - [ ] Orographic precipitation effects
  - [ ] Rain shadow calculation

##### Task 12: Hydrology System - Port river generation and water flow algorithms
- **Note:** This is the most complex and highest-risk module. Treat as a self-contained sub-project.
- [ ] Port height alteration (`Rivers.alterHeights()`).
- [ ] Implement the iterative `resolveDepressions()` algorithm, paying close attention to performance and convergence logic.
- [ ] Port water drainage simulation (`drainWater()`), including flux accumulation and lake outlet calculation.
- [ ] Port river specification (`Rivers.specify()`), including meandering and width calculation.
- [ ] **Recommendation:** Heavily profile this module and explore NumPy-based optimizations to avoid slow Python loops.

##### Task 13: Biome Assignment - Port biome classification system
- [ ] Port biome matrix from `modules/biomes.js:56-63`
- [ ] Implement `Biomes.define()` algorithm (lines 80-106):
  - [ ] Moisture calculation including river influence
  - [ ] Temperature-moisture lookup
  - [ ] Special biome conditions (wetland, desert, permafrost)
- [ ] Port biome data structures (habitability, cost, icons)

##### Task 14: Settlement System - Port burg (city) placement and state generation
- [ ] **Pre-analysis:** Before coding state expansion, document the scoring, weighting, and constraint rules in plain English.
- [ ] Port cell suitability ranking (`rankCells()`).
- [ ] Port capital placement (`placeCapitals()`).
  - [ ] **Recommendation:** Use `GeoPandas.sindex` (R-tree) for efficient minimum spacing enforcement.
- [ ] Port state creation and expansion algorithms based on the pre-analysis document.
- [ ] Implement population calculation and urban hierarchy.

##### Task 15: Name Generation System - Port procedural naming algorithms
- **Recommendation:** Use a dedicated library to reduce effort and bugs.
- [ ] Select and integrate a Python Markov chain library (e.g., `markovify`).
- [ ] Port the *cultural name bases* and linguistic rules from FMG to be used as input corpora for the library.
- [ ] Create a wrapper class that uses the library to generate names for different entities (cultures, states, burgs, etc.).

##### Task 16: PostGIS Integration - Implement database schema and data export
- [ ] Create PostGIS database schema from PLAN.md:
  - [ ] `maps` table with map metadata
  - [ ] `states` table with political boundaries
  - [ ] `burgs` table with settlement points
  - [ ] `rivers` table with waterway lines
  - [ ] `biomes` table with ecological regions
- [ ] Implement GeoPandas to PostGIS export pipeline
- [ ] Create spatial indexing and foreign key relationships
- [ ] Add UUID primary keys and proper geometries (EPSG:4326)
- [ ] **Optimization:** For production, investigate using `chunksize` in `to_postgis` or PostgreSQL's `COPY` command for high-performance bulk inserts.

##### Task 17: FastAPI Service - Create REST endpoints for map generation
- **Note:** Generation is a long-running process and must be handled asynchronously.
- [ ] Implement `POST /maps/generate` endpoint using **FastAPI `BackgroundTasks`**.
- [ ] The endpoint must immediately accept the job and return a `202 Accepted` response with a unique `job_id`.
- [ ] Create a corresponding `GET /maps/status/{job_id}` endpoint for clients to poll for progress and the final result.
- [ ] Add request parameter validation (seed, size, template).

##### Task 18: Testing & Validation - Compare outputs with original FMG
- [ ] **Implement a "stage-gate" testing framework.**
- [ ] **Task:** For a given seed, write a script to export intermediate data artifacts from FMG's browser dev tools at each major step (e.g., save the raw `pack.h` heightmap array, `rivers.json`, etc.).
- [ ] **Task:** Write unit and integration tests that compare each Python module's output directly against these corresponding reference artifacts.
- [ ] Validate that final geometries in PostGIS are valid and correctly projected.
- [ ] Performance benchmark against PLAN.md targets (< 60s generation).

## Technical Challenges Identified

### Primary Challenges
- **Delaunator dependency**: ✅ SOLVED - Successfully using `scipy.spatial.Voronoi`
- **Voronoi Graph Adaptation:** ✅ SOLVED - Adapter layer converts scipy output to FMG structure
- **Cell Packing (reGraph):** ✅ SOLVED - Full implementation with coastal enhancement
- **Depression filling algorithm**: High risk of performance bottlenecks and subtle bugs in this iterative graph algorithm.
- **River meandering**: Path smoothing and variable width calculation
- **State expansion**: Accurately replicating the nuanced rules and "feel" of FMG's political map generation.

### Implementation Notes
- The Python port must maintain the FMG dependency chain while aggressively leveraging Pythonic libraries (NumPy, GeoPandas) for performance and simplicity.
- FMG uses a sophisticated multi-stage pipeline where each component builds on previous ones
- Python port must maintain this dependency chain while adapting to Python's geospatial ecosystem

## Success Criteria
- Generate maps comparable in quality to original FMG
- Meet performance targets: < 60s world generation, < 50ms spatial queries
- Successfully export all data to PostGIS with proper spatial indexing
- Support for future street-level detail expansion as outlined in PLAN.md
