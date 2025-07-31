#!/usr/bin/env python3
"""Test how many hills are actually being added."""

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

# Test config
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "123456789"

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

# Let's check what "5-10" resolves to
print("Testing _get_number_in_range:")
print("-" * 50)

# Reset PRNG for consistency
generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)

for _ in range(10):
    count = generator._get_number_in_range("5-10")
    print(f"  '5-10' -> {count}")

print("\nFirst Hill command: Hill 5-10 15-30 0-30 0-20")
print("This means:")

# Reset PRNG again
generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)

count = int(generator._get_number_in_range("5-10"))
print(f"  Count: {count} hills")

for i in range(count):
    height = generator._get_number_in_range("15-30")
    print(f"  Hill {i+1}: height={height:.0f}")

# Now let's count PRNG calls per individual hill
print("\n\nTesting individual hill PRNG calls:")
print("-" * 50)

# Reset PRNG
generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)
generator._prng.call_count = 0

# Add the first set of hills manually
count = int(generator._get_number_in_range("5-10"))
print(f"Adding {count} hills...")

total_calls = 0
for i in range(count):
    start_calls = generator._prng.call_count
    generator._add_one_hill("15-30", "0-30", "0-20")
    hill_calls = generator._prng.call_count - start_calls
    total_calls += hill_calls

    cells_changed = np.sum(generator.heights > 0)
    print(
        f"  Hill {i+1}: {hill_calls} PRNG calls, total cells with height: {cells_changed}"
    )

print(f"\nTotal PRNG calls for first Hill command: {total_calls + 1}")  # +1 for count
