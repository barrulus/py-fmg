#!/usr/bin/env python3
"""Simple test to verify blob spreading fix."""

import sys
sys.path.append('/home/user/py-fmg')

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng

# Small test
WIDTH = 1200
HEIGHT = 1000  
CELLS_DESIRED = 10000
SEED = "test-fix"

set_random_seed(SEED)

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

generator = HeightmapGenerator(heightmap_config, graph)
prng = get_prng()

# Test single hill
prng.call_count = 0
generator.add_hill("1", "50", "45-55", "45-55")
calls = prng.call_count
land = sum(1 for h in generator.heights if h > 20)

print(f"Single hill test:")
print(f"  PRNG calls: {calls}")
print(f"  Land cells: {land} ({land/len(generator.heights)*100:.1f}%)")
print(f"  Expected: ~20-30 calls for FMG")
print(f"  Ratio: {calls / 25:.1f}x")