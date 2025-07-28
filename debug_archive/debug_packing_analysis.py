#!/usr/bin/env python3
"""
Analyze the packing process to understand the 293-cell difference.
"""

import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph, pack_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def analyze_packing_process():
    """Compare results before and after packing."""
    
    print("🔍 PACKING ANALYSIS - Understanding the 293-cell Gap")
    print("=" * 60)
    
    # Use exact FMG parameters
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    
    # Generate Voronoi and heightmap
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = generator.from_template("lowIsland", main_seed)
    
    print("📊 BEFORE PACKING (Pre-pack Results):")
    print("-" * 40)
    print(f"   Total cells: {len(heights)}")
    print(f"   Land cells: {np.sum(heights >= 20)} ({np.sum(heights >= 20)/len(heights)*100:.1f}%)")
    print(f"   Height range: {np.min(heights)}-{np.max(heights)}")
    print(f"   Mean height: {np.mean(heights):.1f}")
    print()
    
    # Pack the graph (remove deep ocean cells)
    packed_graph = pack_graph(voronoi_graph, heights, deep_ocean_threshold=20)
    
    # Get heights for packed cells only
    keep_mask = heights >= 20
    packed_heights = heights[keep_mask]
    
    print("📦 AFTER PACKING (Post-pack Results):")  
    print("-" * 40)
    print(f"   Total cells: {len(packed_heights)}")
    print(f"   Land cells: {len(packed_heights)} (100.0% - all deep ocean removed)")
    print(f"   Height range: {np.min(packed_heights)}-{np.max(packed_heights)}")
    print(f"   Mean height: {np.mean(packed_heights):.1f}")
    print()
    
    print("🎯 COMPARISON WITH FMG:")
    print("-" * 40)
    print(f"Our packed cells:     {len(packed_heights)}")
    print(f"FMG packed cells:     4499")
    print(f"FMG land cells:       3531")
    print(f"FMG land percentage:  78.5%")
    print()
    
    # Calculate what FMG's land percentage would be if applied to our packed cells
    fmg_land_ratio = 3531 / 4499  # 0.785
    expected_land_cells = int(len(packed_heights) * fmg_land_ratio)
    
    print("🧮 EXPECTED VALUES IF FMG RATIOS APPLIED:")
    print("-" * 40)
    print(f"Expected land cells:  {expected_land_cells}")
    print(f"Our actual land cells: {len(packed_heights)}")
    print(f"Difference:           {len(packed_heights) - expected_land_cells}")
    print()
    
    # Check if we have non-land cells above threshold 20
    cells_20_to_19 = np.sum((heights >= 20) & (heights < 20))  # This should be 0
    print(f"🔍 Cells exactly at threshold 20: {np.sum(heights == 20)}")
    print(f"🔍 Cells 20-29: {np.sum((heights >= 20) & (heights < 30))}")
    print(f"🔍 Cells 30-39: {np.sum((heights >= 30) & (heights < 40))}")
    print(f"🔍 Cells 40+: {np.sum(heights >= 40)}")
    
    return heights, packed_heights, packed_graph

if __name__ == "__main__":
    analyze_packing_process()