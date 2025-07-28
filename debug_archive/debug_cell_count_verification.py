#!/usr/bin/env python3
"""
Verify our cell count matches FMG's expectations and investigate grid parameters.
"""

import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def verify_cell_generation():
    """Verify our cell generation matches FMG's approach."""
    
    print("🔍 CELL COUNT VERIFICATION")
    print("=" * 50)
    
    # Use exact FMG parameters
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    
    print(f"🎯 Configuration:")
    print(f"   Canvas: {config.width}x{config.height}")
    print(f"   Cells desired: {config.cells_desired}")
    
    # Generate Voronoi graph
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    
    print(f"\n📊 Generated Voronoi Graph:")
    print(f"   Actual cells: {len(voronoi_graph.points)}")
    print(f"   Grid cells X: {voronoi_graph.cells_x}")
    print(f"   Grid cells Y: {voronoi_graph.cells_y}")
    print(f"   Calculated cells: {voronoi_graph.cells_x * voronoi_graph.cells_y}")
    
    # Check if our cell count assumption is wrong
    actual_cells = len(voronoi_graph.points)
    if actual_cells != 10000:
        print(f"   ⚠️  DISCREPANCY: Expected 10000, got {actual_cells}")
        
        # Test blob power for actual cell count  
        blob_power_map = {
            1000: 0.93, 2000: 0.95, 5000: 0.97, 10000: 0.98,
            20000: 0.99, 30000: 0.991, 40000: 0.993, 50000: 0.994,
            60000: 0.995, 70000: 0.9955, 80000: 0.996, 90000: 0.9964, 100000: 0.9973
        }
        
        # Find closest blob power for actual cell count
        closest_cell_count = min(blob_power_map.keys(), key=lambda x: abs(x - actual_cells))
        actual_blob_power = blob_power_map.get(actual_cells, blob_power_map[closest_cell_count])
        
        print(f"   Correct blob power for {actual_cells} cells: {actual_blob_power}")
        print(f"   (We were using 0.98 for 10000 cells)")
        
        # Test with correct blob power
        heightmap_config = HeightmapConfig(
            width=300, height=300,
            cells_x=voronoi_graph.cells_x,
            cells_y=voronoi_graph.cells_y,
            cells_desired=actual_cells
        )
        
        generator = HeightmapGenerator(heightmap_config, voronoi_graph)
        generator.blob_power = actual_blob_power
        
        heights = generator.from_template("lowIsland", main_seed)
        land_cells = np.sum(heights >= 20)
        
        print(f"\n🎯 Results with corrected blob power:")
        print(f"   Land cells: {land_cells}")
        print(f"   Target: 3531")
        print(f"   Difference: {land_cells - 3531:+d}")
        
        return actual_cells, actual_blob_power, land_cells
    
    else:
        print(f"   ✅ Cell count matches expectation")
        return actual_cells, 0.98, None

if __name__ == "__main__":
    verify_cell_generation()