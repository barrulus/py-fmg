"""
FMG Settlement and State Generation Analysis

The political system generates capitals, towns, and territorial states with realistic expansion.

## Core Process (modules/burgs-and-states.js)

### Generation Sequence (lines 4-25)
1. `placeCapitals()` - Score-based capital placement with spacing
2. `createStates()` - Initialize states from capitals
3. `placeTowns()` - Add secondary settlements
4. `expandStates()` - Territorial expansion algorithm
5. `normalizeStates()` - Clean up state boundaries
6. `specifyBurgs()` - Calculate settlement properties

### Capital Placement Algorithm (lines 26-69)

**Scoring System** (line 32):
```javascript
score = cells.s * (0.5 + Math.random() * 0.5)  // s = suitability score
```
- Base score from cell suitability ranking
- Random factor (0.5-1.0) for variation
- Higher scores = better capital locations

**Spatial Constraints**:
- Minimum spacing between capitals
- Uses d3.quadtree for efficient spatial queries
- Dynamic spacing reduction if placement fails
- Formula: `spacing = (graphWidth + graphHeight) / 2 / count`

**Requirements**:
- Must have culture assignment (populated)
- Must have positive suitability score
- Must not conflict with existing capitals

### State Creation (lines 72-110)

**State Properties**:
- **Expansionism**: Random factor affecting territorial growth
- **Culture**: Inherited from capital cell culture
- **Name**: Generated from cultural naming patterns
- **Colors**: Assigned for visual distinction
- **Coat of Arms**: Procedurally generated heraldry

### State Expansion Algorithm (lines 284-380)
**Most Complex Political Algorithm**

**Priority Queue System**:
- Uses FlatQueue for efficient expansion
- Each cell has expansion cost calculation
- States grow outward from capitals based on cost

**Cost Calculation** (lines 326-333):
```javascript
cultureCost = culture === cells.culture[e] ? -9 : 100
populationCost = cells.s[e] ? Math.max(20 - cells.s[e], 0) : 5000
biomeCost = getBiomeCost(nativeBiome, cellBiome, stateType)
heightCost = getHeightCost(feature, height, stateType)
riverCost = getRiverCost(riverId, cellId, stateType)
typeCost = getTypeCost(terrainType, stateType)
totalCost = baseCost + cellCost / expansionism
```

**Cost Factors**:
1. **Culture**: -9 for same culture, +100 for foreign
2. **Population**: Lower cost for higher suitability
3. **Biome**: Preference for native biome types
4. **Geography**: Rivers, mountains affect expansion
5. **Expansionism**: State-specific growth modifier

**Expansion Rules**:
- Cannot overwrite locked states
- Cannot overwrite other capitals
- States compete for neutral territory
- Lower total cost = higher expansion priority

### Settlement Hierarchy

**Capital Cities**:
- One per state
- Highest population and importance
- Political and cultural centers
- Fixed positions drive state formation

**Towns/Cities**:
- Secondary settlements within states
- Population-based classification
- Economic and regional centers
- Support state infrastructure

### Geographic Constraints

**Terrain Preferences**:
- Coastal access preferred
- River access valuable
- Mountain barriers limit expansion
- Desert/tundra increase expansion costs

**Cultural Boundaries**:
- Same culture = massive expansion bonus (-9 cost)
- Different culture = major expansion penalty (+100 cost)
- Creates realistic cultural nation-states

### Python Implementation Strategy

1. **Spatial Indexing**:
   ```python
   from sklearn.neighbors import KDTree
   # Replace d3.quadtree with KDTree for spacing checks
   tree = KDTree(existing_capitals)
   distances, indices = tree.query([candidate_position], k=1)
   ```

2. **Priority Queue**:
   ```python
   import heapq
   # Replace FlatQueue with heapq
   expansion_queue = []
   heapq.heappush(expansion_queue, (cost, cell_id, state_id))
   ```

3. **Vectorized Scoring**:
   ```python
   # Calculate all costs simultaneously
   culture_costs = np.where(cell_cultures == state_culture, -9, 100)
   population_costs = np.maximum(20 - cell_suitability, 0)
   total_costs = culture_costs + population_costs + biome_costs + ...
   ```

4. **GeoPandas Integration**:
   ```python
   # Store state boundaries as polygons
   state_polygons = gpd.GeoDataFrame({
       'state_id': state_ids,
       'geometry': [Polygon(boundary_coords) for boundary_coords in boundaries]
   })
   ```

### Critical Implementation Challenges

1. **Cost Function Complexity**: Many interacting factors
2. **Cultural Preferences**: Strong cultural clustering
3. **Geographic Barriers**: Rivers, mountains, biomes
4. **Expansion Competition**: Multiple states growing simultaneously
5. **Realistic Boundaries**: Natural-looking territorial shapes

### Performance Considerations

- **Grid Size Scaling**: O(n log n) with spatial indexing
- **State Count**: More states = more competition
- **Iteration Limits**: Prevent infinite expansion loops
- **Memory Usage**: Large grids need efficient data structures

### Testing and Validation

1. **Cultural Coherence**: States should align with cultural regions
2. **Geographic Logic**: Boundaries should follow natural features
3. **Size Distribution**: Realistic variety in state sizes
4. **Capital Placement**: Good spacing and strategic locations
5. **Visual Comparison**: Generated maps should resemble FMG output

### Key Parameters

- **Spacing Formula**: Based on grid size and state count
- **Expansionism Range**: 1.0 to configurable maximum
- **Growth Rate**: Limits expansion speed
- **Cultural Bonus**: -9 cost for same culture
- **Foreign Penalty**: +100 cost for different culture
- **Population Threshold**: Minimum for settlement placement

### Data Structures Needed

```python
@dataclass
class Settlement:
    id: int
    cell_id: int
    x: float
    y: float
    name: str
    population: int
    is_capital: bool
    state_id: int
    culture_id: int

@dataclass  
class State:
    id: int
    name: str
    capital_id: int
    culture_id: int
    expansionism: float
    color: str
    territory_cells: List[int]
```
"""