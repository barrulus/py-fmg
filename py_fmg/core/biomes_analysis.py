"""
FMG Biome Assignment Analysis

The biome system classifies terrain based on temperature, moisture, and special conditions.

## Core Algorithm (modules/biomes.js)

### Biome Classification Process (lines 80-106)

1. **Input Data**:
   - Temperature: From grid.cells.temp (climate calculation)
   - Precipitation: From grid.cells.prec (weather simulation) 
   - Height: Cell elevation
   - River presence: Boolean from cells.r array

2. **Moisture Calculation** (lines 94-103):
   ```javascript
   moisture = prec[gridReference[cellId]]
   if (riverIds[cellId]) moisture += Math.max(flux[cellId] / 10, 2)
   moistAround = neighbors.map(c => prec[gridReference[c]]).concat([moisture])
   return 4 + mean(moistAround)
   ```
   - Base moisture from precipitation
   - River bonus: flux/10 or minimum +2
   - Neighbor averaging for smooth transitions
   - +4 baseline offset

### Biome Matrix System (lines 56-63)

**5x26 Temperature-Moisture Lookup Table**:
- **Moisture bands**: 0-4 (moisture/5, clamped)
- **Temperature bands**: 0-25 (20-temperature, clamped 0-25)
- Matrix returns biome ID for most cases

**Temperature Ranges**:
- Hot: >19°C (band 0-1)
- Temperate: -4°C to 19°C (band 2-23) 
- Cold: <-4°C (band 24-25)

### Special Biome Conditions (lines 108-125)

**Override Rules** (applied before matrix lookup):
1. **Marine** (ID 0): height < 20 (water)
2. **Permafrost** (ID 11): temperature < -5°C
3. **Hot Desert** (ID 1): temp ≥ 25°C, no river, moisture < 8
4. **Wetland** (ID 12): Complex conditions

**Wetland Rules**:
- Temperature > -2°C (not too cold)
- Near coast: moisture > 40, height < 25
- Inland: moisture > 24, height 25-59

### Biome Properties

```javascript
// Default biome data (lines 6-76)
name: ["Marine", "Hot desert", "Cold desert", "Savanna", "Grassland", 
       "Tropical seasonal forest", "Temperate deciduous forest", 
       "Tropical rainforest", "Temperate rainforest", "Taiga", 
       "Tundra", "Glacier", "Wetland"]

habitability: [0, 4, 10, 22, 30, 50, 100, 80, 90, 12, 4, 0, 12]
cost: [10, 200, 150, 60, 50, 70, 70, 80, 90, 200, 1000, 5000, 150]
```

### Matrix Mapping

**Moisture Band 0 (driest)**:
- Hot: Hot desert (1)
- Cold: Cold desert (2), Tundra (10)

**Moisture Band 1-2 (dry)**:
- Hot: Savanna (3), Grassland (4)  
- Temperate: Temperate deciduous forest (6)
- Cold: Taiga (9), Tundra (10)

**Moisture Band 3-4 (wet)**:
- Hot: Tropical seasonal forest (5)
- Temperate: Temperate rainforest (8)
- Cold: Taiga (9), Tundra (10)

**Highest Moisture + Hot**: Tropical rainforest (7)

### Python Implementation Considerations

1. **NumPy Arrays**:
   ```python
   biome_matrix = np.array([
       [1, 1, 1, ..., 2, 2, 2],  # Moisture band 0
       [3, 3, 3, ..., 10, 10, 10],  # Moisture band 1
       # ... etc
   ], dtype=np.uint8)
   ```

2. **Vectorized Operations**:
   ```python
   # Calculate moisture for all cells
   moisture = precipitation + river_bonus + neighbor_averaging + 4
   
   # Apply special conditions with boolean indexing
   biomes = np.zeros(len(cells), dtype=np.uint8)
   biomes[height < 20] = 0  # Marine
   biomes[temperature < -5] = 11  # Permafrost
   # etc.
   
   # Use matrix for remaining cells
   moisture_bands = np.clip(moisture // 5, 0, 4)
   temp_bands = np.clip(20 - temperature, 0, 25)
   normal_mask = (biomes == 0) & (height >= 20)  # Cells not yet assigned
   biomes[normal_mask] = biome_matrix[moisture_bands[normal_mask], temp_bands[normal_mask]]
   ```

3. **River Influence**:
   - Use scipy.ndimage for neighbor operations
   - Efficient flux-based river bonuses
   - Spatial smoothing for moisture gradients

### Critical Implementation Notes

- **Order matters**: Special conditions must be checked first
- **River bonus**: Significant moisture increase near water
- **Neighbor averaging**: Creates smooth biome transitions
- **Temperature clamping**: Prevents array bounds errors
- **Matrix lookup**: Core classification for most cells
- **Baseline offset**: +4 moisture ensures minimum values

### Testing Strategy

- Compare biome assignments cell-by-cell with FMG
- Validate special condition handling
- Test matrix boundary conditions
- Verify moisture calculation with river influence
- Check neighbor averaging produces smooth transitions

### Performance Optimization

- Single-pass vectorized operations
- Boolean indexing for conditional assignments
- Pre-computed neighbor matrices
- Efficient river detection arrays
"""