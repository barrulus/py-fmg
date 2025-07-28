#!/usr/bin/env python3
"""
Analyze blob power calculation precision and its impact on land generation.
"""

import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def debug_blob_power_precision():
    """Compare different blob power values and their land generation impact."""
    
    print("🔍 BLOB POWER PRECISION ANALYSIS")
    print("=" * 60)
    
    # Use exact FMG parameters
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    
    # Test different blob power values around 0.98
    test_powers = [0.975, 0.98, 0.985, 0.99]
    
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    print("🎯 Testing blob power variations:")
    print("-" * 40)
    
    results = []
    
    for power in test_powers:
        # Create generator with specific blob power
        generator = HeightmapGenerator(heightmap_config, voronoi_graph)
        
        # Manually set blob power for testing
        original_power = generator.blob_power
        generator.blob_power = power
        
        # Generate heightmap
        heights = generator.from_template("lowIsland", main_seed)
        
        land_cells = np.sum(heights >= 20)
        mean_height = np.mean(heights)
        max_height = np.max(heights)
        
        results.append({
            'power': power,
            'land_cells': land_cells,
            'mean_height': mean_height,
            'max_height': max_height
        })
        
        print(f"Power {power:.3f}: {land_cells:4d} land cells (mean: {mean_height:.1f}, max: {max_height})")
        
        # Restore original power
        generator.blob_power = original_power
    
    print()
    print("🎯 FMG Target: 3531 land cells")
    print("-" * 30)
    
    # Find closest result to target
    target = 3531
    closest = min(results, key=lambda x: abs(x['land_cells'] - target))
    
    print(f"Closest match: Power {closest['power']:.3f} → {closest['land_cells']} cells")
    print(f"Difference from target: {closest['land_cells'] - target:+d} cells")
    
    # Check if any power gives exactly 3531
    exact_matches = [r for r in results if r['land_cells'] == target]
    if exact_matches:
        print(f"Exact match found: Power {exact_matches[0]['power']:.3f}")
    else:
        print("No exact match found in tested range")
    
    print()
    print("💡 Analysis:")
    print("-" * 20)
    
    # Calculate sensitivity
    power_range = max(test_powers) - min(test_powers)
    cell_range = max(r['land_cells'] for r in results) - min(r['land_cells'] for r in results)
    sensitivity = cell_range / power_range
    
    print(f"Blob power sensitivity: {sensitivity:.0f} cells per 0.001 power change")
    
    # Estimate required power for exact match
    current_power = 0.98
    current_cells = next(r['land_cells'] for r in results if r['power'] == current_power)
    cell_difference = current_cells - target
    
    estimated_power = current_power - (cell_difference / sensitivity * 0.001)
    print(f"Estimated power for 3531 cells: {estimated_power:.4f}")
    
    return results

if __name__ == "__main__":
    debug_blob_power_precision()