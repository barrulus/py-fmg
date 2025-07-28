#!/usr/bin/env python3
"""
Detailed debugging of heightmap generation to identify compatibility gaps.
"""

import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def debug_heightmap_step_by_step():
    """Debug heightmap generation with detailed logging."""
    
    print("🔍 DETAILED HEIGHTMAP DEBUGGING")
    print("=" * 60)
    
    # Use exact FMG parameters
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    
    print(f"🎯 Target FMG Results:")
    print(f"   - Height range: 0-51, Mean: 26.6")
    print(f"   - Land cells: 3531/4499 (78.5%)")
    print(f"   - Packed cells: 4499")
    print()
    
    # Step 1: Generate Voronoi Graph
    print("📊 STEP 1: Voronoi Graph Generation")
    print("-" * 40)
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    print(f"✅ Generated {len(voronoi_graph.points)} cells")
    
    # Check first few cell coordinates
    print(f"🔍 First 5 cell coordinates:")
    for i in range(5):
        x, y = voronoi_graph.points[i]
        print(f"   Cell {i}: ({x:.6f}, {y:.6f})")
    print()
    
    # Step 2: Initialize Heightmap Generator
    print("🏔️  STEP 2: Heightmap Generator Setup")
    print("-" * 40)
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    print(f"✅ Initialized for {len(generator.heights)} cells")
    print(f"🔍 Blob power: {generator.blob_power}")
    print(f"🔍 Line power: {generator.line_power}")
    print()
    
    # Execute template
    heights = generator.from_template("lowIsland", main_seed)
    
    # Final results
    print("🎯 FINAL RESULTS COMPARISON")
    print("-" * 40)
    
    print(f"Our Results:")
    print(f"   - Height range: {np.min(heights)}-{np.max(heights)}")
    print(f"   - Mean height: {np.mean(heights):.1f}")
    print(f"   - Land cells: {np.sum(heights >= 20)}/{len(heights)} "
          f"({np.sum(heights >= 20)/len(heights)*100:.1f}%)")
    
    print(f"\nFMG Target:")
    print(f"   - Height range: 0-51")
    print(f"   - Mean height: 26.6")  
    print(f"   - Land cells: 3531/4499 (78.5%)")
    
    return heights

if __name__ == "__main__":
    debug_heightmap_step_by_step()