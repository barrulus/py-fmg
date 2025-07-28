# FMG Heightmap Generation Process Analysis

## Key Findings

### 1. Seed Management Flow

1. **Initial seed setting** (`main.js:setSeed()`):
   - If no seed provided: generates new seed via `generateSeed()` which uses `Math.random()` 
   - Sets `Math.random = aleaPRNG(seed)` to use deterministic PRNG

2. **Grid generation** (`generateMap()` -> `shouldRegenerateGrid()`):
   - Checks if grid needs regeneration based on:
     - Seed mismatch: `expectedSeed !== grid.seed`
     - Cell count change
     - Grid dimensions change
   - If regeneration needed: calls `generateGrid()` which:
     - **RESETS THE PRNG**: `Math.random = aleaPRNG(seed)`
     - Stores seed in grid object: `return {..., seed}`

3. **Heightmap generation** (`HeightmapGenerator.generate()`):
   - **RESETS PRNG AGAIN**: `Math.random = aleaPRNG(seed)`
   - Uses template or precreated heightmap

### 2. The Seed Problem

From the FMG console output:
```
Seed: 854906727
Canvas size: 1200x799 px
Heightmap: fractious
Template: random template
```

But in our saved map data, we see:
- Original seed from URL/generation: "651658815"
- Grid was reused from this seed
- But heightmap generation used: "854906727"

### 3. Critical Issue: Grid Reuse

When `shouldRegenerateGrid()` returns false (grid can be reused):
- The grid still has the OLD seed stored in `grid.seed`
- But the global `seed` variable has the NEW seed
- Heightmap generation uses the global `seed`, not `grid.seed`

### 4. Process Flow

```
1. User loads map with seed "651658815"
2. Grid generated with seed "651658815" and stored
3. User regenerates map
4. New seed "854906727" generated
5. shouldRegenerateGrid() returns false (same dimensions)
6. Grid reused with old seed "651658815" 
7. Heightmap generated with new seed "854906727"
8. Mismatch between grid seed and heightmap seed!
```

### 5. Template Selection

The "random template" selection happens in heightmap generation:
- When template is "random", FMG randomly selects from available templates
- This selection uses the current PRNG state
- The actual template used is not stored/logged

### 6. Function Call Sequence

1. `generateMap()`
2. `setSeed()` - sets global seed and PRNG
3. `shouldRegenerateGrid()` - checks if grid needs regeneration
4. If no regeneration: reuses existing grid (with old seed!)
5. `HeightmapGenerator.generate()` - resets PRNG with global seed
6. `fromTemplate()` - uses current PRNG for random selections
7. `reGraph()` - packs cells based on heightmap

## Conclusion

The core issue is that FMG can reuse a grid from a previous seed while generating heightmap with a new seed. This creates a mismatch between the Voronoi structure (old seed) and heightmap values (new seed).

To replicate FMG behavior exactly, we need to:
1. Track both grid seed and heightmap seed separately
2. Allow grid reuse when dimensions match
3. Use the correct seed for each operation