"""
FMG Voronoi Diagram Generation Analysis

The Voronoi diagram system in FMG creates a grid-based foundation for all map generation.

## Key Components (utils/graphUtils.js + modules/voronoi.js)

### Grid Generation Process (graphUtils.js)

1. **Point Placement**
   - `placePoints()`: Creates jittered square grid
   - Spacing calculated from desired cell count: sqrt((width*height)/cellsDesired)
   - Grid dimensions: cellsX, cellsY calculated from spacing

2. **Jittered Grid (getJitteredGrid, lines 82-97)**
   - Regular square grid with randomized positions
   - Jittering: ±45% of spacing (radius * 0.9)
   - Prevents artificial regular patterns
   - Formula: x + (random() * 2*jittering - jittering)

3. **Boundary Points (getBoundaryPoints, lines 59-79)**
   - Additional points around map edges for clipping
   - Prevents infinite Voronoi cells at borders
   - Spacing: 2x normal spacing
   - Creates pseudo-boundary for finite diagram

4. **Delaunay Triangulation**
   - Uses Mapbox Delaunator library
   - Input: jittered points + boundary points
   - Output: triangles (vertices) and halfedges (connectivity)

### Voronoi Construction (modules/voronoi.js)

1. **Voronoi Class Constructor (lines 9-36)**
   - Converts Delaunay triangulation to dual Voronoi diagram
   - Each triangle circumcenter becomes Voronoi vertex
   - Each point becomes Voronoi cell
   - Builds connectivity graph between cells

2. **Cell Data Structure**
   ```javascript
   cells.v[p] = adjacent_vertices  // Voronoi vertices bounding cell
   cells.c[p] = adjacent_cells     // Neighboring cells
   cells.b[p] = is_border         // Border cell flag
   ```

3. **Vertex Data Structure**
   ```javascript
   vertices.p[t] = coordinates     // Vertex position (triangle circumcenter)
   vertices.v[t] = adjacent_vertices // Neighboring vertices
   vertices.c[t] = adjacent_cells   // Adjacent cells
   ```

### Key Algorithms

1. **Half-edge Navigation**
   - `edgesAroundPoint()`: Finds all edges touching a point
   - `nextHalfedge()`, `prevHalfedge()`: Triangle edge navigation
   - Enables efficient neighbor discovery

2. **Circumcenter Calculation (lines 122-135)**
   - Circumcenter of triangle = Voronoi vertex
   - Mathematical formula from triangle vertices
   - Creates natural cell boundaries

3. **Border Detection**
   - Border cells have fewer neighbors than edges
   - Used for coastline and boundary handling

### Python Port Considerations

1. **Use scipy.spatial.Voronoi**
   - Replace Delaunator + custom Voronoi class
   - scipy provides: vertices, ridge_vertices, ridge_points
   - Need adapter to match FMG's cell-centric structure

2. **Critical Adapter Requirements**
   - Convert scipy output to cell.c (neighbors) format
   - Maintain vertex ordering for consistent geometry
   - Handle boundary clipping equivalent to FMG
   - Preserve border cell detection

3. **Data Structure Mapping**
   ```python
   # FMG format needed:
   cells.c[i] = [neighbor_cell_ids]     # Adjacent cells
   cells.v[i] = [vertex_ids]            # Cell vertices (ordered)
   cells.b[i] = is_border               # Border flag
   
   # From scipy.spatial.Voronoi:
   vor.ridge_points  # Cell pairs sharing edge
   vor.vertices      # Vertex coordinates
   vor.ridge_vertices # Vertex pairs forming ridges
   ```

4. **Performance Optimizations**
   - NumPy arrays for coordinates and connectivity
   - Spatial indexing for neighbor queries
   - Vectorized operations where possible

### Grid Parameters
- Standard sizes: 1000-100000+ cells
- Typical spacing: 10-50 units
- Jittering: 45% of spacing
- Boundary offset: -1 * spacing
- Boundary spacing: 2 * normal spacing

### Critical Implementation Notes
- Point ordering must be consistent with original grid
- Boundary handling affects coastline generation
- Cell connectivity is fundamental to all algorithms
- Border detection crucial for geographic features
- Vertex ordering affects polygon rendering

### Testing Strategy
- Compare cell connectivity graphs between FMG and Python
- Validate vertex coordinates match expected positions
- Ensure border detection produces same results
- Test with various grid sizes and seeds
"""