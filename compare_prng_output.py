#!/usr/bin/env python3
"""
Compare PRNG output and point generation between Python and FMG.
"""

import json
from py_fmg.core.alea_prng import AleaPRNG

def analyze_fmg_export():
    """Analyze the FMG export to understand the pattern."""
    with open('voronoi-data-162921633(1).json', 'r') as f:
        fmg_data = json.load(f)
    
    points = fmg_data['voronoi']['cells']['p']
    print(f"FMG Export Analysis (seed: {fmg_data['seed']}):")
    print(f"Total cells: {len(points)}")
    print(f"First 10 points:")
    for i in range(min(10, len(points))):
        print(f"  {i}: {points[i]}")
    
    # Find bounds
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    print(f"\nBounds:")
    print(f"  X: {min_x:.2f} to {max_x:.2f}")
    print(f"  Y: {min_y:.2f} to {max_y:.2f}")
    
    # Check if points start from corners
    corner_points = [p for p in points if (p[0] < 50 and p[1] < 50)]
    print(f"\nPoints near top-left corner (< 50,50): {len(corner_points)}")
    if corner_points:
        print("First few corner points:")
        for p in corner_points[:5]:
            print(f"  {p}")

def test_grid_generation():
    """Test our grid generation to see if we can match FMG."""
    seed = "162921633"
    width = 1200
    height = 1000
    
    # Calculate spacing like FMG
    cells_desired = 10000
    spacing = ((width * height) / cells_desired) ** 0.5
    print(f"\nPython Grid Generation (seed: {seed}):")
    print(f"Spacing: {spacing}")
    
    # Generate a few points manually
    prng = AleaPRNG(seed)
    radius = spacing / 2
    jittering = radius * 0.9
    double_jittering = jittering * 2
    
    print(f"Radius: {radius}")
    print(f"Jittering: {jittering}")
    
    # Try to understand FMG's pattern
    # FMG might be using a different grid generation approach
    print("\nChecking if FMG uses packed grid from start...")
    
    # The cell count suggests FMG might be using a different algorithm
    actual_cells = 5821
    actual_spacing = ((width * height) / actual_cells) ** 0.5
    print(f"If targeting {actual_cells} cells, spacing would be: {actual_spacing}")

if __name__ == "__main__":
    analyze_fmg_export()
    test_grid_generation()