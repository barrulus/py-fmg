# connectVertices analysis

The connectVertices algorithm in /utils/pathUtils.js is a graph traversal algorithm that traces the boundary between regions of different types in a Voronoi
diagram. Here's a detailed analysis:

## Algorithm Purpose

Creates a continuous chain of vertices that forms the boundary between cells of the same type and cells of different types. This is used to generate isolines
(contour lines) for features like coastlines, political borders, or terrain boundaries.

## Key Components

### Input Parameters

- vertices: The Voronoi graph's vertex data structure
- startingVertex: Initial vertex on the boundary
- ofSameType: Function to check if a cell matches the target type
- addToChecked: Optional function to mark cells as processed
- closeRing: Whether to close the path by adding the starting vertex at the end

### Core Logic (lines 145-174)

The algorithm walks along the boundary by examining each vertex's three neighboring cells and vertices:

const [c1, c2, c3] = neibCells.map(ofSameType);  // Check which cells match type
const [v1, v2, v3] = vertices.v[current];// Get neighboring vertices

// Choose next vertex based on boundary conditions
if (v1 !== previous && c1 !== c2) next = v1;
else if (v2 !== previous && c2 !== c3) next = v2;
else if (v3 !== previous && c1 !== c3) next = v3;

### Decision Rules

The algorithm selects the next vertex by finding where the cell types change:
- c1 !== c2: Boundary exists between cells 1 and 2, so follow edge to vertex 1
- c2 !== c3: Boundary exists between cells 2 and 3, so follow edge to vertex 2
- c1 !== c3: Boundary exists between cells 1 and 3, so follow edge to vertex 3

The v !== previous check prevents backtracking.

### Termination Conditions

1. Success: Returns to starting vertex (forms complete loop)
2. Error cases:
  - Next vertex out of bounds
  - No valid next vertex found
  - Maximum iterations exceeded (prevents infinite loops)

## Usage Context

The algorithm is primarily used in getIsolines to:
1. Find all boundaries between different region types
2. Generate paths for rendering (SVG paths, polygons)
3. Create special effects like halos or water gaps

Example Application

For a coastline:
- "Same type" = land cells (height ≥ 20)
- "Different type" = water cells (height < 20)
- Result: Chain of vertices forming the exact coastline

The algorithm efficiently traces complex, irregular boundaries in O(n) time where n is the number of boundary vertices.


## Usage in features.js

The connectVertices function is called on line 209 within the getFeatureVertices function:

return connectVertices({vertices, startingVertex, ofSameType, closeRing: false});

### Context and Purpose

The function is used during the markupPack process to trace the perimeter vertices of geographic features (islands and lakes). Here's the call chain:

1. markupPack() - Main function that identifies and marks all features
2. addFeature() - Creates feature objects with properties
3. getCellsData() - Determines feature type and finds boundary data
4. getFeatureVertices() - Uses connectVertices to trace the feature boundary

### Key Implementation Details

#### Feature Identification Process (lines 181-211)

1. Find Starting Cell: Locates a cell on the feature boundary
const isOnBorder = cellId => borderCells[cellId] || neighbors[cellId].some(ofDifferentType);
2. Find Starting Vertex: Identifies a vertex on the type boundary
const startingVertex = cells.v[startCell].find(v => vertices.c[v].some(ofDifferentType));
3. Trace Boundary: Uses connectVertices with:
  - closeRing: false - Doesn't close the path (handled later)
  - ofSameType - Function checking if cell belongs to the feature
  - No addToChecked - Feature cells already marked during flood fill

#### Feature Types and Processing

The traced vertices are used differently based on feature type:

- Islands: Vertices define the coastline
- Lakes:
  - Vertices define the lake boundary
  - Direction corrected if needed (line 174)
  - Shoreline cells extracted from vertices (line 175)
- Oceans: No vertex tracing needed (line 182)

#### Integration with Feature Properties

The vertex chain is used to calculate:
- Area: Using d3.polygonArea() on vertex positions
- Perimeter: The vertices themselves form the perimeter
- Shoreline: For lakes, cells adjacent to the vertices

#### Differences from pathUtils.js Usage <<- TODO analyse this against the python port >>

In features.js, connectVertices is used:
- Without addToChecked: Cells already marked during feature detection
- Without closeRing: Feature detection handles closure separately
- For single features: Each call traces one specific feature
- During initial markup: Part of the world generation pipeline

This demonstrates how the same boundary-tracing algorithm serves multiple purposes - from generating isolines for rendering to defining exact feature boundaries
during world generation.
