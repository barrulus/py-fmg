#!/usr/bin/env python3
"""Test single hill generation to debug PRNG call count."""

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

# Set up test parameters
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "123456789"

# Generate graph
config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

# Create heightmap generator with proper seeding
heightmap_config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)
prng = generator._prng
prng.call_count = 0

# Add just one hill
print("Adding single hill...")
start_count = prng.call_count

# Test different scenarios
test_cases = [
    ("1", "50", "40-60", "40-60"),  # Single hill, fixed height
    ("5", "50", "40-60", "40-60"),  # 5 hills, fixed height
    ("5-10", "50", "40-60", "40-60"),  # Random count, fixed height
    ("5", "40-60", "40-60", "40-60"),  # Fixed count, random height
]

for count, height, range_x, range_y in test_cases:
    # Reset generator for consistent testing
    generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)
    prng = generator._prng
    prng.call_count = 0

    generator.add_hill(count, height, range_x, range_y)

    calls = prng.call_count
    land = sum(1 for h in generator.heights if h > 20)
    land_pct = land / len(generator.heights) * 100

    print(f"\nTest: count={count}, height={height}")
    print(f"  PRNG calls: {calls}")
    print(f"  Land: {land_pct:.1f}%")

    # Check specific call patterns
    if count == "1" and height == "50":
        print("\nDetailed analysis for single fixed hill:")
        # We expect:
        # 1. _get_number_in_range for count (0 calls if "1")
        # 2. _get_number_in_range for height (0 calls if "50")
        # 3. _get_point_in_range for x (1 call)
        # 4. _get_point_in_range for y (1 call)
        # 5. Blob spreading (many calls)
        print(f"  Expected minimum: 2 (for x,y positioning)")
        print(f"  Actual: {calls}")
