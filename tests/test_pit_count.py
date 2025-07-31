#!/usr/bin/env python3
"""Test PRNG calls for pits."""

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

# Test config
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "123456"

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

# Initialize heights to non-water values so pits can be placed
generator.heights = np.full(generator.n_cells, 50.0, dtype=np.float32)

print("Testing _add_one_pit PRNG calls:")
print("-" * 50)
print(f"Blob power: {generator.blob_power}")
print(f"Total cells: {generator.n_cells}")
print()

# Get PRNG from generator
prng = generator._prng
prng.call_count = 0

# Test individual pits
for i in range(5):
    # Reset heights
    generator.heights = np.full(generator.n_cells, 50.0, dtype=np.float32)

    start_calls = prng.call_count
    generator._add_one_pit("20-30", "40-60", "40-60")
    pit_calls = prng.call_count - start_calls

    # Count cells affected (those with height < 50)
    cells_affected = np.sum(generator.heights < 50)
    min_height = np.min(generator.heights)

    print(
        f"Pit {i+1}: {pit_calls} PRNG calls, {cells_affected} cells affected, min height: {min_height:.1f}"
    )

print("\n\nComparison:")
print("-" * 50)
print("_add_one_hill: ~18,000 PRNG calls, ~9,788 cells affected")
print("_add_one_pit:  Expecting ~30-50 PRNG calls, ~30-50 cells affected")

# Let's also test with the actual trough command from isthmus
print("\n\nTesting trough command from isthmus template:")
print("-" * 50)

# Reset for isthmus test
generator = HeightmapGenerator(
    heightmap_config, graph, seed="654321"
)  # Same seed as isthmus
prng = generator._prng
prng.call_count = 0
generator.heights = np.full(generator.n_cells, 50.0, dtype=np.float32)

# First trough command: "Trough 4-8 15-30 0-30 0-20"
count = int(generator._get_number_in_range("4-8"))
print(f"Trough count: {count}")

total_trough_calls = 0
for i in range(count):
    start_calls = prng.call_count
    generator._add_one_trough("15-30", "0-30", "0-20", None, None)
    trough_calls = prng.call_count - start_calls
    total_trough_calls += trough_calls
    print(f"  Trough {i+1}: {trough_calls} PRNG calls")

print(
    f"\nTotal PRNG calls for trough command: {total_trough_calls + 1}"
)  # +1 for count

print("\n\nAnalysis:")
print("-" * 50)
print("The pit/trough functions use smooth float decay (h gets smaller each iteration)")
print("This causes natural termination when h < 1")
print("The hill function uses integer truncation which causes decay to stall")
print("This results in massive plateaus that spread across the entire map")
