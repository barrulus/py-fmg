#!/usr/bin/env python3
"""Test hill generation with detailed PRNG tracking."""

import sys
sys.path.append('/home/user/py-fmg')

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng

# Test parameters
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "test-hill"

# Set random seed
set_random_seed(SEED)

# Generate graph
config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

# Create heightmap generator  
heightmap_config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

generator = HeightmapGenerator(heightmap_config, graph)

# Get PRNG
prng = get_prng()

# Test _get_number_in_range
print("Testing _get_number_in_range:")
print("-" * 40)
test_values = ["1", "5", "10", "5-10", "15-30", "0.5", "0.25"]
for val in test_values:
    prng.call_count = 0
    result = generator._get_number_in_range(val)
    print(f"  {val:10s} -> {result:6.1f} (PRNG calls: {prng.call_count})")

# Test _get_point_in_range
print("\nTesting _get_point_in_range:")
print("-" * 40)
test_ranges = ["0-100", "40-60", "0-30", "70-100"]
for range_str in test_ranges:
    prng.call_count = 0
    result = generator._get_point_in_range(range_str, 1000)
    print(f"  {range_str:10s} -> {result:6.1f} (PRNG calls: {prng.call_count})")

# Now test actual hill generation step by step
print("\nTesting hill generation step by step:")
print("-" * 40)

# Manually trace through add_hill for "1" "50" "40-60" "40-60"
prng.call_count = 0

# Step 1: Get count
count_val = generator._get_number_in_range("1")
print(f"1. Count: {count_val} (PRNG calls: {prng.call_count})")

# Step 2: For one hill, get height
prev_count = prng.call_count
height_val = generator._get_number_in_range("50")
print(f"2. Height: {height_val} (PRNG calls: {prng.call_count - prev_count})")

# Step 3: Get position
prev_count = prng.call_count
x_val = generator._get_point_in_range("40-60", WIDTH)
print(f"3. X position: {x_val} (PRNG calls: {prng.call_count - prev_count})")

prev_count = prng.call_count
y_val = generator._get_point_in_range("40-60", HEIGHT)
print(f"4. Y position: {y_val} (PRNG calls: {prng.call_count - prev_count})")

# Step 4: Find starting cell
start_cell = generator._find_grid_cell(x_val, y_val)
print(f"5. Start cell: {start_cell} (no PRNG calls)")

# Step 5: Count blob spreading calls
print(f"\nTotal PRNG calls before blob spreading: {prng.call_count}")

# Now run the actual add_hill and see the difference
print("\nRunning actual add_hill:")
print("-" * 40)
generator.heights[:] = 0
prng.call_count = 0

generator.add_hill("1", "50", "40-60", "40-60")

print(f"Total PRNG calls: {prng.call_count}")
land = sum(1 for h in generator.heights if h > 20)
print(f"Land cells: {land}")

# Check if the issue is with repeated attempts
print("\nTesting with impossible constraints:")
print("-" * 40)
generator.heights[:] = 90  # Make all cells high
prng.call_count = 0

generator.add_hill("1", "50", "40-60", "40-60")

print(f"Total PRNG calls with high terrain: {prng.call_count}")
generator.heights[:] = 0  # Reset