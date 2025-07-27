"""
FMG Heightmap Generation Analysis

The heightmap generation system in FMG uses a template-based approach with procedural algorithms.

## Core Algorithms (modules/heightmap-generator.js)

### Template System
- String-based commands parsed and executed sequentially
- Commands: Hill, Pit, Range, Trough, Strait, Mask, Invert, Add, Multiply, Smooth
- Parameters include count, height, X/Y ranges

### Key Algorithms:

1. **addHill() (lines 129-163)**
   - Blob spreading algorithm using power-based decay
   - Creates hills with exponential height falloff
   - Uses grid.cells.c for neighbor connectivity
   - Formula: change[c] = change[q] ** blobPower * (Math.random() * 0.2 + 0.9)
   - blobPower varies by grid size (0.93-0.9973)

2. **addPit() (lines 165-199)**
   - Similar to hill but subtracts height
   - Prevents creation below sea level (height < 20)
   - Uses same blob spreading with power decay

3. **addRange() (lines 202-292)**
   - Creates mountain ranges between start/end points
   - Uses pathfinding to create ridge line
   - Spreads height from ridge outward
   - Uses linePower for decay along range (0.75-0.93)
   - Adds prominences every 6th cell along ridge

4. **addTrough() (lines 294+)**
   - Creates valleys/depressions
   - Similar pathfinding to Range but subtracts height

5. **addStrait()**
   - Creates water channels cutting through land
   - Can be vertical or horizontal

### Power Factors
- **blobPower**: Controls hill/pit spread (0.93-0.9973 based on grid size)
- **linePower**: Controls range spread (0.75-0.93 based on grid size)
- Higher values = more concentrated features
- Larger grids use higher power values for realistic scaling

### Height Management
- Heights stored as Uint8Array (0-100)
- Sea level at height < 20
- Maximum height 100
- lim() function clamps values to valid range

### Template Examples (config/heightmap-templates.js)
```
volcano:
Hill 1 90-100 44-56 40-60    # Single high peak
Multiply 0.8 50-100 0 0      # Reduce high elevations
Range 1.5 30-55 45-55 40-60  # Add ridges
Smooth 3 0 0 0               # Smooth terrain
```

### NumPy Optimization Opportunities
1. Replace iterative blob spreading with vectorized distance-based calculations
2. Use scipy.ndimage for smoothing operations
3. Implement pathfinding with vectorized operations
4. Use boolean indexing for masking operations

### Key Data Structures
- heights: Uint8Array of elevation values per cell
- grid.cells.c: Cell connectivity (neighbors for each cell)
- grid.points: [x,y] coordinates for each cell
- change arrays: Temporary height modifications during generation

### Critical Implementation Notes
- Blob spreading uses breadth-first queue traversal
- Random factors maintain terrain variation
- Power factors must scale with grid size for realistic features
- Height clamping prevents overflow/underflow
- Sea level threshold (20) is hardcoded but configurable
"""