#!/usr/bin/env python3
"""
Test the multiply operation specifically.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed


def main():
    # Create small test map
    config = GridConfig(width=30, height=30, cells_desired=100)
    graph = generate_voronoi_graph(config, seed="test")
    
    heightmap_config = HeightmapConfig(
        width=30,
        height=30,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=100,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Set some test heights
    generator.heights[:] = 30  # All cells at height 30
    generator.heights[:20] = 10  # First 20 cells at height 10 (water)
    
    print(f"Before multiply:")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    print(f"  First 10 heights: {generator.heights[:10]}")
    
    # Apply multiply operation: Multiply 0.4 20-100
    # This should multiply all land cells (20-100) by 0.4
    generator.modify("20-100", multiply=0.4)
    
    print(f"\nAfter multiply 0.4 on range 20-100:")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    print(f"  First 10 heights: {generator.heights[:10]}")
    print(f"  Heights 20-30: {generator.heights[20:30]}")
    
    # The land cells (30) should become 30 * 0.4 = 12
    # But FMG might handle it differently for land cells
    
    # Test with the actual lowIsland template operations
    print("\n\n=== Testing lowIsland sequence ===")
    
    # Reset
    set_random_seed("651658815")
    generator2 = HeightmapGenerator(heightmap_config, graph)
    
    # Simulate having done the hills/ranges/troughs
    # Set up a height distribution similar to what we'd expect
    generator2.heights[:] = 0
    generator2.heights[30:70] = 40  # Some land in the middle
    generator2.heights[40:60] = 50  # Higher land
    
    print(f"\nBefore Multiply 0.4 20-100:")
    print(f"  Heights: min={generator2.heights.min()}, max={generator2.heights.max()}, mean={generator2.heights.mean():.1f}")
    print(f"  Land percentage: {np.sum(generator2.heights >= 20) / len(generator2.heights) * 100:.1f}%")
    
    # Apply the multiply from lowIsland template
    generator2.modify("20-100", multiply=0.4)
    
    print(f"\nAfter Multiply 0.4 20-100:")
    print(f"  Heights: min={generator2.heights.min()}, max={generator2.heights.max()}, mean={generator2.heights.mean():.1f}")
    print(f"  Land percentage: {np.sum(generator2.heights >= 20) / len(generator2.heights) * 100:.1f}%")
    print(f"  Sample of modified cells: {generator2.heights[40:50]}")


if __name__ == "__main__":
    main()