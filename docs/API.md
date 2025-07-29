# py-fmg API Documentation

## Core Module

### Voronoi Graph Generation

#### `generate_voronoi_graph(config, seed=None, apply_relaxation=True)`

Generate a complete Voronoi graph matching FMG structure.

**Parameters:**
- `config` (GridConfig): Grid configuration with width, height, and cells_desired
- `seed` (str, optional): Random seed for reproducibility
- `apply_relaxation` (bool): Whether to apply Lloyd's relaxation (default: True)

**Returns:**
- `VoronoiGraph`: Complete Voronoi graph data structure

**Example:**
```python
from py_fmg.core import GridConfig, generate_voronoi_graph

config = GridConfig(width=1000, height=1000, cells_desired=10000)
graph = generate_voronoi_graph(config, seed="test123", apply_relaxation=True)
```

#### `generate_or_reuse_grid(existing_grid, config, seed=None, apply_relaxation=True)`

Generate new grid or reuse existing one based on parameters. Enables the "keep land, reroll mountains" workflow.

**Parameters:**
- `existing_grid` (VoronoiGraph, optional): Previously generated grid
- `config` (GridConfig): Grid configuration
- `seed` (str, optional): Random seed
- `apply_relaxation` (bool): Whether to apply Lloyd's relaxation

**Returns:**
- `VoronoiGraph`: Either new or reused grid

**Example:**
```python
# First generation
grid = generate_voronoi_graph(config, "myseed")

# Keep same landmasses, but allow different heightmap
grid = generate_or_reuse_grid(grid, config, "myseed")  # Reuses existing
```

### Cell Packing (reGraph)

#### `regraph(graph)`

Perform FMG's reGraph operation to create a packed cell structure. Reduces ~10,000 cells to ~4,500 by removing deep ocean while enhancing coastlines.

**Parameters:**
- `graph` (VoronoiGraph): Original VoronoiGraph with heights calculated

**Returns:**
- `VoronoiGraph`: New packed VoronoiGraph with fewer cells and enhanced coastlines

**Example:**
```python
from py_fmg.core import regraph

# After heightmap generation
packed = regraph(graph)
print(f"Reduced from {len(graph.points)} to {len(packed.points)} cells")
```

### Heightmap Generation

#### `HeightmapGenerator(config, graph)`

Generates heightmaps using various procedural algorithms.

**Parameters:**
- `config` (HeightmapConfig): Heightmap configuration
- `graph` (VoronoiGraph): Voronoi graph structure

**Methods:**
- `add_hill(count, height, range_x, range_y)`: Add hills using blob spreading
- `add_pit(count, height, range_x, range_y)`: Add depressions
- `add_range(count, height, range_x, range_y)`: Add mountain ranges
- `add_trough(count, height, range_x, range_y)`: Add valleys
- `add_strait(width, direction)`: Add water channels
- `smooth(factor, add)`: Smooth heightmap by averaging with neighbors
- `mask(power)`: Apply radial mask (fade edges)
- `from_template(template_name, seed)`: Generate from named template

**Example:**
```python
from py_fmg.core import HeightmapGenerator, HeightmapConfig

hm_config = HeightmapConfig(
    width=1000, height=1000,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=10000,
    spacing=graph.spacing
)

generator = HeightmapGenerator(hm_config, graph)
heights = generator.from_template("fractious", "seed123")
```

## Data Structures

### GridConfig

Configuration for grid generation.

```python
GridConfig(
    width: float,      # Map width
    height: float,     # Map height  
    cells_desired: int # Target number of cells
)
```

### VoronoiGraph

Mutable dataclass containing complete Voronoi graph data.

**Attributes:**
- `spacing`: Grid spacing
- `cells_desired`: Target cell count
- `graph_width`, `graph_height`: Original generation dimensions
- `seed`: Generation seed
- `points`: Cell center coordinates
- `heights`: Pre-allocated height values
- `cell_neighbors`: Adjacency lists
- `cell_vertices`: Vertex lists per cell
- `cell_border_flags`: Border cell indicators
- `grid_indices`: Mapping for packed cells (set by regraph)

**Methods:**
- `should_regenerate(config, seed)`: Check if regeneration needed

### CellType

Cell type classification constants:

```python
CellType.INLAND = 0          # Land cell with no water neighbors
CellType.LAND_COAST = 1      # Land cell with water neighbors
CellType.WATER_COAST = -1    # Water cell with land neighbors
CellType.LAKE = -2           # Lake cell
CellType.DEEP_OCEAN = -3     # Deep ocean (excluded by reGraph)
```

## Complete Workflow Example

```python
from py_fmg.core import (
    GridConfig, HeightmapConfig, 
    generate_voronoi_graph, HeightmapGenerator, regraph
)

# 1. Generate Voronoi graph
config = GridConfig(width=1000, height=1000, cells_desired=10000)
graph = generate_voronoi_graph(config, seed="map123", apply_relaxation=True)

# 2. Generate heightmap
hm_config = HeightmapConfig(
    width=1000, height=1000,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=10000,
    spacing=graph.spacing
)

generator = HeightmapGenerator(hm_config, graph)
graph.heights[:] = generator.from_template("fractious", "terrain456")

# 3. Pack cells for performance (remove deep ocean)
packed = regraph(graph)

# Result: packed graph ready for cultures, states, rivers, etc.
print(f"Original cells: {len(graph.points)}")
print(f"Packed cells: {len(packed.points)}")
print(f"Reduction: {(1 - len(packed.points)/len(graph.points))*100:.1f}%")
```

## Key Features

1. **Lloyd's Relaxation**: Improves point distribution by ~42%
2. **Height Pre-allocation**: Heights array created during grid generation
3. **Grid Reuse**: Keep landmasses while changing terrain
4. **Cell Packing**: Reduces cell count by ~55% for performance
5. **Coastal Enhancement**: Adds detail along coastlines
6. **Full FMG Compatibility**: Matches original JavaScript behavior