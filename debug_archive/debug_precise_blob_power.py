#!/usr/bin/env python3
"""
Find the exact blob power that produces 3531 land cells.
"""

import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def find_exact_blob_power():
    """Binary search to find blob power that produces exactly 3531 land cells."""
    
    print("🔍 PRECISE BLOB POWER SEARCH")
    print("=" * 50)
    
    # Use exact FMG parameters
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    target_cells = 3531
    
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    # Test very fine-grained powers around 0.98
    test_powers = [
        0.9790, 0.9795, 0.9798, 0.9799,
        0.9800, 0.9801, 0.9802, 0.9805
    ]
    
    print("🎯 Fine-grained blob power search:")
    print("-" * 40)
    
    results = []
    
    for power in test_powers:
        generator = HeightmapGenerator(heightmap_config, voronoi_graph)
        generator.blob_power = power
        
        heights = generator.from_template("lowIsland", main_seed)
        land_cells = np.sum(heights >= 20)
        
        results.append((power, land_cells))
        diff = land_cells - target_cells
        
        print(f"Power {power:.4f}: {land_cells:4d} cells (Δ{diff:+4d})")
        
        # If we found exact match, highlight it
        if land_cells == target_cells:
            print(f"  *** EXACT MATCH FOUND! ***")
    
    print()
    print("🎯 Target: 3531 cells")
    print("-" * 20)
    
    # Find closest matches
    closest = min(results, key=lambda x: abs(x[1] - target_cells))
    print(f"Closest: Power {closest[0]:.4f} → {closest[1]} cells (Δ{closest[1] - target_cells:+d})")
    
    # Check for exact matches
    exact = [r for r in results if r[1] == target_cells]
    if exact:
        print(f"Exact matches: {len(exact)} found")
        for power, cells in exact:
            print(f"  Power {power:.4f} → {cells} cells")
    else:
        print("No exact matches found")
        
        # Find the range that brackets the target
        under = [r for r in results if r[1] < target_cells]
        over = [r for r in results if r[1] > target_cells]
        
        if under and over:
            best_under = max(under, key=lambda x: x[1])
            best_over = min(over, key=lambda x: x[1])  
            
            print(f"Target range: {best_under[0]:.4f} ({best_under[1]}) < target < {best_over[0]:.4f} ({best_over[1]})")
            
            # Linear interpolation estimate
            power_diff = best_over[0] - best_under[0]
            cell_diff = best_over[1] - best_under[1]
            target_offset = target_cells - best_under[1]
            
            estimated_power = best_under[0] + (target_offset / cell_diff) * power_diff
            print(f"Interpolated estimate: {estimated_power:.6f}")
    
    return results

if __name__ == "__main__":
    find_exact_blob_power()