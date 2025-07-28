#!/usr/bin/env python3
"""
Trace through FMG's exact logic to understand the cell count difference.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template


def trace_fmg_logic():
    """Trace FMG's exact filtering logic."""
    
    print("🔍 TRACING FMG'S EXACT LOGIC")
    print("=" * 50)
    
    # Setup
    grid_seed = "651658815"
    map_seed = "651658815"
    
    # Generate Voronoi and heightmap
    config = GridConfig(width=300, height=300, cells_desired=10000)
    voronoi_graph = generate_voronoi_graph(config, grid_seed)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000,
        spacing=voronoi_graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = generator.from_template("lowIsland", map_seed)
    
    print(f"Starting with {len(heights)} grid cells")
    
    # FMG's logic from reGraph():
    # if (height < 20 && type !== -1 && type !== -2) continue;
    
    # The KEY insight: FMG doesn't include ALL land cells!
    # It's using Features.markupGrid() which does something more complex
    
    print("\n📊 UNDERSTANDING THE DIFFERENCE:")
    print("-" * 40)
    
    # Count cells by height
    land_cells = np.sum(heights >= 20)
    ocean_cells = np.sum(heights < 20)
    
    print(f"Land cells (h >= 20): {land_cells}")
    print(f"Ocean cells (h < 20): {ocean_cells}")
    
    # But FMG only gets ~4264 cells total!
    # This means it's excluding ~5736 cells
    
    # Let's think about this differently...
    # What if FMG's Features.markupGrid() assigns types that cause many cells to be excluded?
    
    print("\n💡 HYPOTHESIS:")
    print("-" * 40)
    print("FMG's Features.markupGrid() must be doing something that causes")
    print("many land cells to be excluded during reGraph.")
    print("")
    print("Possibilities:")
    print("1. Some land cells are marked as type that gets excluded")
    print("2. The 'features' system groups cells and only keeps representatives")
    print("3. There's additional filtering we're not seeing")
    
    # Let's check the distribution more carefully
    print("\n📊 HEIGHT DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    # Look at the exact counts
    for h in range(15, 52):
        count = np.sum(heights == h)
        if count > 0:
            print(f"Height {h}: {count} cells")
    
    # The real answer might be in how FMG generates the initial heightmap
    # Maybe it's not starting with 10,000 cells at all for packing?
    
    print("\n🔍 CRITICAL INSIGHT:")
    print("-" * 40)
    print("Wait... what if the issue is earlier?")
    print("What if FMG's heightmap generation creates a different distribution")
    print("that naturally results in fewer cells after filtering?")
    
    # Check our ocean distribution
    ocean_distribution = []
    for i in range(len(heights)):
        if heights[i] < 20:
            ocean_distribution.append(heights[i])
    
    ocean_distribution = np.array(ocean_distribution)
    print(f"\nOcean cells (h < 20): {len(ocean_distribution)}")
    print(f"Ocean height range: {np.min(ocean_distribution)} - {np.max(ocean_distribution)}")
    
    # Key question: why do we have so few ocean cells?
    print("\n❓ KEY QUESTIONS:")
    print("-" * 40)
    print("1. Why do we only have 839 ocean cells vs FMG's ~1057?")
    print("2. Why is our minimum height 15 instead of 2?")
    print("3. Are we generating the heightmap correctly?")
    
    # The answer: we need to fix the heightmap generation first!
    print("\n✅ CONCLUSION:")
    print("-" * 40)
    print("The issue is NOT in reGraph filtering!")
    print("The issue is in our heightmap generation creating too much land.")
    print("We need to fix the minimum height issue (15 vs 2) first.")


if __name__ == "__main__":
    trace_fmg_logic()