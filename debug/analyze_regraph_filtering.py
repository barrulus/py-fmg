#!/usr/bin/env python3
"""
Analyze exactly what's happening in reGraph filtering.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.regraph import _get_cell_type
from py_fmg.config.heightmap_templates import get_template


def analyze_filtering():
    """Analyze the filtering process in detail."""
    
    print("🔍 REGRAPH FILTERING ANALYSIS")
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
    
    print(f"Total grid cells: {len(heights)}")
    print(f"Height range: {np.min(heights)} - {np.max(heights)}")
    
    # Analyze cell types
    print("\n📊 CELL TYPE ANALYSIS:")
    print("-" * 40)
    
    type_counts = {-3: 0, -2: 0, -1: 0, 0: 0, 1: 0}
    kept_cells = 0
    excluded_cells = 0
    
    # Track height distribution of excluded cells
    excluded_heights = []
    
    for i in range(len(voronoi_graph.points)):
        height = heights[i]
        cell_type = _get_cell_type(i, heights, voronoi_graph)
        type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
        
        # Apply FMG's filtering logic
        if height < 20 and cell_type not in [-1, -2]:
            excluded_cells += 1
            excluded_heights.append(height)
        else:
            kept_cells += 1
    
    print("\nCell type distribution:")
    print("Type -3 (deep ocean): ", type_counts.get(-3, 0))
    print("Type -2 (lake):       ", type_counts.get(-2, 0))
    print("Type -1 (water coast):", type_counts.get(-1, 0))
    print("Type  0 (inland):     ", type_counts.get(0, 0))
    print("Type  1 (land coast): ", type_counts.get(1, 0))
    
    print(f"\nKept cells: {kept_cells}")
    print(f"Excluded cells: {excluded_cells}")
    
    # Analyze height distribution of cells
    print("\n📊 HEIGHT DISTRIBUTION OF CELLS:")
    print("-" * 40)
    
    # Count cells by height range
    height_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
    
    print("All cells:")
    for min_h, max_h in height_ranges:
        count = np.sum((heights >= min_h) & (heights < max_h))
        print(f"  {min_h:2d}-{max_h:3d}: {count:5d}")
    
    print("\nExcluded cells height distribution:")
    if excluded_heights:
        excluded_heights = np.array(excluded_heights)
        for min_h, max_h in height_ranges:
            count = np.sum((excluded_heights >= min_h) & (excluded_heights < max_h))
            if count > 0:
                print(f"  {min_h:2d}-{max_h:3d}: {count:5d}")
    
    # The issue: we should be getting ~4264 cells, not 9829
    print("\n💡 ANALYSIS:")
    print("-" * 40)
    print(f"FMG gets ~4,264 cells from 10,000")
    print(f"We're getting {kept_cells} cells")
    print(f"Difference: {kept_cells - 4264}")
    
    # Check how many cells have height < 20
    ocean_cells = np.sum(heights < 20)
    print(f"\nTotal cells with h < 20: {ocean_cells}")
    print(f"Ocean cells we're keeping: {np.sum(heights < 20) - excluded_cells}")
    
    # The real issue might be that we're not properly identifying ocean vs coast
    print("\n🔍 COASTLINE ANALYSIS:")
    print("-" * 40)
    
    # Count actual coastline cells (cells with both land and water neighbors)
    true_coast_count = 0
    for i in range(len(voronoi_graph.points)):
        has_water_neighbor = False
        has_land_neighbor = False
        
        for neighbor in voronoi_graph.cell_neighbors[i]:
            if heights[neighbor] < 20:
                has_water_neighbor = True
            else:
                has_land_neighbor = True
        
        if has_water_neighbor and has_land_neighbor:
            true_coast_count += 1
    
    print(f"True coastline cells: {true_coast_count}")
    
    # The problem: FMG uses Features.markupGrid() to properly classify cells
    # We need to implement this classification step!
    print("\n❌ ROOT CAUSE:")
    print("-" * 40)
    print("FMG uses Features.markupGrid() to classify cells BEFORE reGraph")
    print("This sets cell types based on geographic features, not just height")
    print("We're missing this crucial step!")


if __name__ == "__main__":
    analyze_filtering()