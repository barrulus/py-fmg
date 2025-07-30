#!/usr/bin/env python3
"""Test single hill generation to debug PRNG call count."""

import sys
sys.path.append('/home/user/py-fmg')

from py_fmg.core.voronoi_graph import VoronoiGraph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed
from py_fmg.core.alea_prng import AleaPRNG

# Set up test parameters
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "test-single-hill"

# Set random seed
set_random_seed(SEED)

# Create graph
graph = VoronoiGraph(
    width=WIDTH,
    height=HEIGHT,
    cells_desired=CELLS_DESIRED,
    seed=SEED,
)

# Generate graph
voronoi_data = graph.generate()

# Create heightmap generator
config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

generator = HeightmapGenerator(config, voronoi_data)

# Reset call counter
from py_fmg.utils.random import get_prng
prng = get_prng()
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
    # Reset heights and counter
    generator.heights[:] = 0
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