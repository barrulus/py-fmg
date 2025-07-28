#!/usr/bin/env python3
"""
Debug mask coordinate calculation.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph


def main():
    # Generate grid
    config = GridConfig(width=300, height=300, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed="1234567")
    
    print(f"Grid dimensions: {graph.cells_x}x{graph.cells_y}")
    print(f"Spacing: {graph.spacing}")
    print(f"Number of points: {len(graph.points)}")
    
    # Check corner cells
    corner_indices = [
        0,  # top-left
        graph.cells_x - 1,  # top-right
        graph.cells_x * (graph.cells_y - 1),  # bottom-left
        graph.cells_x * graph.cells_y - 1  # bottom-right
    ]
    
    print("\nCorner cell coordinates:")
    for i, idx in enumerate(corner_indices):
        x, y = graph.points[idx]
        print(f"  Cell {idx}: ({x:.2f}, {y:.2f})")
    
    # Check expected vs actual
    print("\nExpected corner coordinates:")
    print(f"  Top-left: (1.5, 1.5)")
    print(f"  Top-right: ({graph.spacing * (graph.cells_x - 1) + graph.spacing/2:.1f}, 1.5)")
    print(f"  Bottom-left: (1.5, {graph.spacing * (graph.cells_y - 1) + graph.spacing/2:.1f})")
    print(f"  Bottom-right: ({graph.spacing * (graph.cells_x - 1) + graph.spacing/2:.1f}, {graph.spacing * (graph.cells_y - 1) + graph.spacing/2:.1f})")
    
    # Test mask distance calculation
    print("\n\nMask distance calculation for corners:")
    width, height = 300, 300
    
    for i, idx in enumerate(corner_indices):
        x, y = graph.points[idx]
        nx = 2 * x / width - 1
        ny = 2 * y / height - 1
        distance = (1 - nx**2) * (1 - ny**2)
        
        print(f"  Cell {idx}: x={x:.2f}, y={y:.2f}")
        print(f"    nx={nx:.4f}, ny={ny:.4f}")
        print(f"    distance={(1 - nx**2) * (1 - ny**2):.6f}")
        
        # What height would this give from 50?
        power = 4
        fr = power
        h = 50
        masked = h * distance
        result = (h * (fr - 1) + masked) / fr
        print(f"    50 -> {result:.1f}")


if __name__ == "__main__":
    main()