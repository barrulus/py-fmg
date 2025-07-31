# ConnectVertices Algorithm Analysis

## Overview

The `connectVertices` algorithm is a graph traversal algorithm in Azgaar's Fantasy Map Generator that traces boundaries between different region types in a Voronoi diagram. It's a fundamental building block for boundary detection and rendering throughout the application.

## Algorithm Purpose

Creates a continuous chain of vertices forming the boundary between cells of the same type and cells of different types. Used for generating:
- Coastlines
- Political borders
- Biome boundaries
- Cultural/religious boundaries
- Lake perimeters
- Any type-based region boundaries

## Algorithm Details

### Input Parameters

```javascript
connectVertices({vertices, startingVertex, ofSameType, addToChecked, closeRing})
```

- `vertices`: The Voronoi graph's vertex data structure
- `startingVertex`: Initial vertex on the boundary (must be provided)
- `ofSameType`: Function to check if a cell matches the target type
- `addToChecked`: Optional function to mark cells as processed
- `closeRing`: Whether to close the path by adding the starting vertex at the end

### Core Logic

The algorithm walks along the boundary by examining each vertex's three neighboring cells and vertices:

```javascript
const [c1, c2, c3] = neibCells.map(ofSameType);  // Check which cells match type
const [v1, v2, v3] = vertices.v[current];        // Get neighboring vertices

// Choose next vertex based on boundary conditions
if (v1 !== previous && c1 !== c2) next = v1;      // Boundary between cells 1 and 2
else if (v2 !== previous && c2 !== c3) next = v2; // Boundary between cells 2 and 3
else if (v3 !== previous && c1 !== c3) next = v3; // Boundary between cells 1 and 3
```

### Decision Rules

The algorithm selects the next vertex by finding where cell types change:
- `c1 !== c2`: Type boundary exists between cells 1 and 2 → follow edge to vertex 1
- `c2 !== c3`: Type boundary exists between cells 2 and 3 → follow edge to vertex 2
- `c1 !== c3`: Type boundary exists between cells 1 and 3 → follow edge to vertex 3

The `v !== previous` check prevents backtracking.

### Termination Conditions

1. **Success**: Returns to starting vertex (forms complete loop)
2. **Error cases**:
   - Next vertex out of bounds
   - No valid next vertex found
   - Maximum iterations exceeded (prevents infinite loops)

## Finding the Starting Point

`connectVertices` requires a starting vertex, which is found through a two-step process:

### Step 1: Find a Boundary Cell

```javascript
function findOnBorderCell(firstCell) {
  const isOnBorder = cellId => 
    borderCells[cellId] || neighbors[cellId].some(ofDifferentType);
  
  if (isOnBorder(firstCell)) return firstCell;
  
  const startCell = cells.i.filter(ofSameType).find(isOnBorder);
  return startCell;
}
```

### Step 2: Find a Boundary Vertex

```javascript
function getFeatureVertices(startCell) {
  const startingVertex = cells.v[startCell].find(
    v => vertices.c[v].some(ofDifferentType)
  );
  return connectVertices({vertices, startingVertex, ofSameType, closeRing: false});
}
```

## Usage Contexts

### 1. Feature Detection (features.js)

Used during `markupPack()` to trace perimeters of geographic features:

```
markupPack() 
  └─> addFeature() 
      └─> getCellsData() 
          └─> getFeatureVertices() 
              └─> connectVertices()
```

- Traces island coastlines
- Defines lake boundaries
- Calculates feature areas and perimeters

### 2. Rendering System (layers.js via pathUtils.js)

Used by `getIsolines()` for rendering various boundaries:

- **Biome borders**: Natural region boundaries
- **Culture borders**: Cultural group territories
- **Religion borders**: Religious influence areas
- **Political borders**: State boundaries with optional halos
- **Province borders**: Administrative divisions
- **Selected cells**: User selection outlines

### 3. Analysis Tools

- `getPolesOfInaccessibility()`: Finds optimal label positions within features
- `getVertexPath()`: Creates paths for non-continuous cell arrays

## Call Hierarchy

```
User Actions / Generation Pipeline
    │
    ├─> World Generation
    │   └─> markupPack()
    │       └─> connectVertices()
    │
    ├─> Layer Rendering
    │   └─> getIsolines()
    │       └─> connectVertices()
    │
    └─> Cell Selection
        └─> getVertexPath()
            └─> connectVertices()
```

## Trigger Events

### User-Initiated
- Toggling layer visibility
- Editing heightmap
- Regenerating map features
- Selecting cells
- Modifying political/cultural entities

### Automatic Processes
- Initial map generation
- Feature recalculation after edits
- Border rendering updates
- Real-time preview updates

## Performance Characteristics

- **Time Complexity**: O(n) where n is the number of boundary vertices
- **Space Complexity**: O(n) for the vertex chain storage
- **Efficiency**: Traces complex boundaries in a single pass
- **Scalability**: Handles irregular boundaries of any complexity

## Example Applications

### Coastline Generation
```javascript
// Land = cells with height ≥ 20
const ofSameType = cellId => cells.h[cellId] >= 20;
const coastline = connectVertices({vertices, startingVertex, ofSameType});
```

### Political Border with Water Gaps
```javascript
const isolines = getIsolines(pack, cellId => cells.state[cellId], {
  fill: true,
  waterGap: true,  // Skip water sections
  halo: true       // Add border emphasis
});
```

## Implementation Notes

1. **Deterministic**: Always produces the same result for the same input
2. **Direction-aware**: Traces boundaries consistently (counterclockwise for islands)
3. **Error-tolerant**: Includes safeguards against infinite loops and invalid states
4. **Modular**: Separation of concerns between finding start points and tracing boundaries
5. **Flexible**: Supports various options through the calling functions (fill, halo, waterGap)

The `connectVertices` algorithm is essential to FMG's ability to generate and render complex geographic and political boundaries efficiently and accurately.