#!/usr/bin/env python3
"""
Test multiply and mask operations in isolation.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed


def main():
    # Generate the same grid as FMG
    config = GridConfig(width=300, height=300, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed="1234567")
    
    heightmap_config = HeightmapConfig(
        width=300,
        height=300,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=10000,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Test 1: Set heights to simulate post-hills state
    print("Test 1: Multiply operation on land cells")
    print("=" * 60)
    
    # Set up test heights - all land
    generator.heights[:] = 50  # All cells at height 50
    generator.heights[:1000] = 30  # Some lower land
    generator.heights[:100] = 20  # Some barely land
    
    print("Before multiply:")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    
    # Apply multiply 0.4 to range 20-100
    generator.modify("20-100", multiply=0.4)
    
    print("\nAfter multiply 0.4 on 20-100:")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    
    # Check specific transformations
    print("\nExpected transformations:")
    print("  50 -> (50-20)*0.4+20 = 32")
    print("  30 -> (30-20)*0.4+20 = 24")
    print("  20 -> (20-20)*0.4+20 = 20")
    
    # Test 2: Apply mask
    print("\n\nTest 2: Mask operation")
    print("=" * 60)
    
    # Reset heights
    generator.heights[:] = 32  # Post-multiply height
    
    print("Before mask:")
    print(f"  All heights = 32")
    
    # Apply mask with power 4
    generator.mask(4)
    
    print("\nAfter mask(4):")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    
    # Check corner values
    corner_indices = [0, 99, 9900, 9999]
    print("\nCorner heights:")
    for idx in corner_indices:
        print(f"  Cell {idx}: {generator.heights[idx]}")
    
    # Test 3: Combined effect
    print("\n\nTest 3: Combined multiply then mask")
    print("=" * 60)
    
    # Start with typical post-hill heights
    generator.heights[:] = 0
    # Add a circular pattern to simulate hills
    for i in range(len(generator.heights)):
        x, y = graph.points[i]
        dist_from_center = np.sqrt((x - 150)**2 + (y - 150)**2)
        height = max(0, 100 - dist_from_center * 0.5)
        generator.heights[i] = int(height)
    
    print("Initial pattern (simulated hills):")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    
    # Apply multiply
    generator.modify("20-100", multiply=0.4)
    
    print("\nAfter multiply 0.4:")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    
    # Apply mask
    generator.mask(4)
    
    print("\nAfter mask(4):")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)}")
    
    # Show corner values
    print("\nFinal corner heights:")
    for idx in corner_indices:
        print(f"  Cell {idx}: {generator.heights[idx]}")


if __name__ == "__main__":
    main()