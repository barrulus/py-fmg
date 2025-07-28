#!/usr/bin/env python3
"""
Test mask operation to understand the difference.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig


def main():
    # Create test heightmap
    config = GridConfig(width=100, height=100, cells_desired=100)
    graph = generate_voronoi_graph(config, seed="test")
    
    heightmap_config = HeightmapConfig(
        width=100,
        height=100,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=100,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Set all heights to 50
    generator.heights[:] = 50
    
    print("Before mask:")
    print(f"  All heights: {generator.heights.min()} to {generator.heights.max()}")
    
    # Apply mask with power 4
    generator.mask(4)
    
    print("\nAfter mask(4):")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    
    # Show spatial pattern
    heights_2d = generator.heights.reshape(graph.cells_y, graph.cells_x)
    
    print(f"\nCorner values:")
    print(f"  Top-left: {heights_2d[0, 0]}")
    print(f"  Top-right: {heights_2d[0, -1]}")
    print(f"  Bottom-left: {heights_2d[-1, 0]}")
    print(f"  Bottom-right: {heights_2d[-1, -1]}")
    print(f"  Center: {heights_2d[5, 5]}")
    
    print(f"\nFull grid:")
    print(heights_2d)
    
    # Test distance calculation manually
    print("\n\nManual distance calculation test:")
    width, height = 100, 100
    
    # Test corner points
    test_points = [
        (0, 0, "top-left"),
        (99, 0, "top-right"),
        (0, 99, "bottom-left"),
        (99, 99, "bottom-right"),
        (50, 50, "center")
    ]
    
    for x, y, label in test_points:
        nx = 2 * x / width - 1
        ny = 2 * y / height - 1
        distance = (1 - nx**2) * (1 - ny**2)
        print(f"  {label}: x={x}, y={y} -> nx={nx:.2f}, ny={ny:.2f} -> distance={distance:.4f}")


if __name__ == "__main__":
    main()