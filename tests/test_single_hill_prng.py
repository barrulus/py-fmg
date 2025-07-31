#!/usr/bin/env python3
"""Test PRNG calls for a single hill."""

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

# Small test
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "test-single-hill"

config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

heightmap_config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

# Pass seed to HeightmapGenerator for proper PRNG initialization
generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)
prng = generator._prng
prng.call_count = 0

print(f"Blob power: {generator.blob_power}")
print(f"Total cells: {generator.n_cells}")
print()

# Add ONE hill and track PRNG calls
print("Adding one hill with height=20, range_x='50-50', range_y='50-50'")
prng.call_count = 0
generator.add_hill(1, 20, "50-50", "50-50")

print(f"PRNG calls for one hill: {prng.call_count}")
print(f"Cells with height > 0: {np.sum(generator.heights > 0)}")
print(f"Max height: {np.max(generator.heights)}")

# Check the spreading pattern
unique_heights = np.unique(generator.heights[generator.heights > 0])
print(f"\nUnique height values: {len(unique_heights)}")
print(f"Height distribution: {unique_heights[:10]}... {unique_heights[-10:]}")

# Let's trace through manually what should happen
print("\n\nManual calculation of expected PRNG calls:")
print("-" * 50)

# Simulate spreading
h = 20
spread_cells = 1  # Start cell
prng_calls = 0
avg_neighbors = 6

print(f"Start: h={h}")

# Track how many cells at each height
height_counts = {h: 1}

while h > 1:
    h_float = (h ** 0.98) * 1.0  # Average multiplier
    h_next = int(h_float)
    
    if h_next == h:
        print(f"  h={h} -> {h_float:.2f} -> {h_next} (STALLS!)")
        # In FMG, this creates a plateau
        # All remaining cells get the same height
        plateau_cells = spread_cells * avg_neighbors * 10  # Rough estimate
        prng_calls += plateau_cells
        print(f"  Creates plateau with ~{plateau_cells} cells")
        break
    else:
        print(f"  h={h} -> {h_float:.2f} -> {h_next}")
        # Each cell spreads to ~6 neighbors
        new_cells = spread_cells * avg_neighbors
        prng_calls += new_cells
        spread_cells = new_cells
        
        if h_next not in height_counts:
            height_counts[h_next] = 0
        height_counts[h_next] += new_cells
        
    h = h_next

print(f"\nExpected PRNG calls: ~{prng_calls}")
print(f"Actual PRNG calls: {prng.call_count}")

# Check for plateau effect
height_counts_actual = {}
for h in unique_heights:
    count = np.sum(generator.heights == h)
    height_counts_actual[int(h)] = count

print("\nHeight value distribution:")
for h in sorted(height_counts_actual.keys(), reverse=True)[:10]:
    print(f"  Height {h}: {height_counts_actual[h]} cells")