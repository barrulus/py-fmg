"""
FMG Hydrology and River Generation Analysis

The hydrology system is the most complex module in FMG, handling water flow simulation,
depression filling, and river formation.

## Core Process (modules/river-generator.js)

### Generation Sequence (lines 22-36)
1. `alterHeights()` - Modify heightmap for water flow
2. `Lakes.detectCloseLakes(h)` - Identify lake systems
3. `resolveDepressions(h)` - Fill depressions iteratively
4. `drainWater()` - Simulate water flow and river formation
5. `defineRivers()` - Create final river segments with properties
6. `calculateConfluenceFlux()` - Calculate flow at river junctions
7. `downcutRivers()` - Erode river beds (optional)

### Depression Resolution Algorithm (lines 257-313)
**Critical Algorithm - High Risk for Bugs**

- **Purpose**: Eliminate local minima that would trap water
- **Method**: Iterative elevation of lowest points
- **Max Iterations**: Configurable (default ~100)
- **Process**:
  1. Sort land cells by elevation (lowest first)
  2. For each cell, check if it's lower than all neighbors
  3. If depressed, raise to min_neighbor_height + 0.1
  4. Handle lakes separately with different elevation increment (+0.2)
  5. Continue until no depressions remain or max iterations reached

**Implementation Notes**:
- **Performance Critical**: Can be O(n²) in worst case
- **Convergence Issues**: May fail with complex terrain
- **Lake Handling**: Lakes can be elevated or closed if persistent
- **Progress Tracking**: Aborts if no improvement detected

### Water Drainage Simulation (lines 38-125)
**Core Water Flow Algorithm**

1. **Precipitation Input** (line 47)
   ```javascript
   cells.fl[i] += prec[cells.g[i]] / cellsNumberModifier
   ```
   - Add precipitation flux to each land cell
   - Scale by grid size modifier

2. **Lake Outlets** (lines 49-83)
   - Calculate lake evaporation vs inflow
   - Create outlets for lakes with positive balance
   - Handle chain lakes (connected lake systems)
   - Assign tributary rivers to outlet basins

3. **Downhill Flow** (lines 88-124)
   - Find lowest neighbor for each cell
   - Check minimum flux threshold (30) for river formation
   - Create new rivers when flux exceeds threshold
   - Call `flowDown()` to propagate water

### Flow Propagation (lines 127-161)
**River Network Formation**

- **Confluence Logic**: When rivers meet, larger flux takes precedence
- **River Assignment**: Cells assigned to dominant river
- **Tributary Tracking**: Parent-child relationships maintained
- **Lake Interaction**: Rivers flow into lakes, lakes have outlets

### River Specification (lines 163-220)
**Final River Properties**

1. **Width Calculation**: Based on discharge and distance from source
2. **Meandering**: Add intermediate points for natural river curves
3. **Length Calculation**: Approximate length from meandered path
4. **Discharge**: Final water volume at river mouth

### Key Data Structures

```javascript
cells.fl[i]   // Water flux (m³/s) - accumulated precipitation + upstream flow
cells.r[i]    // River ID assigned to cell (0 = no river)
cells.conf[i] // Confluence marker (where rivers meet)
cells.h[i]    // Cell height (modified during generation)

riversData[riverID] = [cell1, cell2, ...]  // River path as cell sequence
riverParents[riverID] = parentRiverID       // Tributary relationships
```

### Critical Parameters

- **Sea Level**: height < 20 (hardcoded water threshold)
- **Min River Flux**: 30 (minimum flow to form visible river)
- **Confluence Detection**: Tracks where rivers merge
- **Meandering Factor**: 0.5 + variations for natural curves
- **Width Factors**: Scale with grid size for realistic appearance

### NumPy Optimization Opportunities

1. **Depression Resolution**: 
   - Use scipy.ndimage.label for connected components
   - Vectorized neighbor finding with array indexing
   - Priority queues for efficient processing

2. **Flow Accumulation**:
   - Vectorized precipitation addition
   - Efficient downhill routing with array operations
   - Batch confluence detection

3. **River Tracing**:
   - NumPy-based pathfinding algorithms
   - Vectorized width/discharge calculations

### Performance Considerations

- **Grid Size Scaling**: Algorithms scale poorly with large grids
- **Depression Count**: More depressions = longer resolution time
- **River Complexity**: Dense river networks increase computation
- **Memory Usage**: Multiple arrays scale with grid size

### Testing Strategy

- **Reference Data**: Export intermediate flux, height, and river arrays from FMG
- **Stage Validation**: Compare after each major step
- **Convergence Testing**: Verify depression resolution completes
- **Flow Conservation**: Ensure water balance maintains
- **Visual Comparison**: River networks should match FMG output

### Implementation Risks

1. **Depression Algorithm**: Most likely to have performance/convergence issues
2. **Confluence Logic**: Complex river merging rules
3. **Lake Handling**: Special cases for closed basins
4. **Floating Point Precision**: Height comparisons sensitive to rounding
"""