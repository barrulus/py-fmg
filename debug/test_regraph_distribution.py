#!/usr/bin/env python3
"""
Test the complete pipeline with reGraph to verify final distribution.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.regraph import regraph
from py_fmg.config.heightmap_templates import get_template


def test_regraph_distribution():
    """Test the complete generation pipeline with reGraph."""
    
    print("🔍 REGRAPH DISTRIBUTION TEST")
    print("=" * 50)
    
    # Setup - match FMG test case
    grid_seed = "651658815"
    map_seed = "651658815"
    
    # Stage 1: Generate initial Voronoi graph (10,000 points)
    print("\n📊 STAGE 1: Initial Voronoi Generation")
    print("-" * 40)
    
    config = GridConfig(width=300, height=300, cells_desired=10000)
    voronoi_graph = generate_voronoi_graph(config, grid_seed)
    
    print(f"Generated {len(voronoi_graph.points)} Voronoi points")
    print(f"Grid: {voronoi_graph.cells_x}x{voronoi_graph.cells_y}")
    print(f"Spacing: {voronoi_graph.spacing}")
    
    # Stage 2: Generate heightmap
    print("\n📊 STAGE 2: Heightmap Generation")
    print("-" * 40)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000,
        spacing=voronoi_graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = generator.from_template("lowIsland", map_seed)
    
    print(f"Height range: {np.min(heights)} - {np.max(heights)}")
    print(f"Cells with h >= 20: {np.sum(heights >= 20)}")
    
    # Stage 3: Perform reGraph coastal resampling
    print("\n📊 STAGE 3: ReGraph Coastal Resampling")
    print("-" * 40)
    
    regraph_result = regraph(voronoi_graph, heights, config, grid_seed)
    
    print(f"Original points: {len(voronoi_graph.points)}")
    print(f"After filtering deep ocean: {len(regraph_result.points)}")
    print(f"New Voronoi cells: {len(regraph_result.voronoi_graph.points)}")
    
    # Analyze the reGraphed data
    final_heights = regraph_result.heights
    final_graph = regraph_result.voronoi_graph
    
    # Calculate statistics on the final packed cells
    land_cells = np.sum(final_heights >= 20)
    ocean_cells = np.sum(final_heights < 20)
    total_cells = len(final_heights)
    
    print("\n🎯 FINAL DISTRIBUTION:")
    print("-" * 40)
    print(f"Total packed cells: {total_cells}")
    print(f"Land cells (h≥20): {land_cells} ({land_cells/total_cells*100:.1f}%)")
    print(f"Ocean cells (h<20): {ocean_cells} ({ocean_cells/total_cells*100:.1f}%)")
    print(f"Height range: {np.min(final_heights)}-{np.max(final_heights)}")
    print(f"Mean height: {np.mean(final_heights):.1f}")
    
    # Height distribution
    print("\n📊 HEIGHT DISTRIBUTION:")
    print("-" * 40)
    ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
    
    print("Range    | Count")
    print("---------|------")
    for min_h, max_h in ranges:
        count = np.sum((final_heights >= min_h) & (final_heights < max_h))
        print(f"{min_h:2d}-{max_h:3d}   | {count:5d}")
    
    # FMG Reference for comparison
    print("\n📊 FMG REFERENCE (seed 651658815):")
    print("-" * 40)
    print("Total packed cells: 4,264")
    print("Land cells (h≥20): 3,207 (75.2%)")
    print("Height range: 2-51")
    print("Mean height: 26.7")
    print("\nHeight Distribution:")
    print("Range    | FMG")
    print("---------|-----")
    print(" 0-10    |   18")
    print("10-20    | 1039")
    print("20-30    | 1823")
    print("30-40    |  907")
    print("40-50    |  318")
    print("50-100   |  159")
    
    # Analyze differences
    print("\n💡 ANALYSIS:")
    print("-" * 40)
    
    # Check minimum height cells
    min_height_cells = np.sum(final_heights < 5)
    print(f"Cells with height < 5: {min_height_cells}")
    
    # Check coastline cells
    coastline_count = 0
    for i in range(len(final_graph.points)):
        if i >= len(final_heights):
            break
        cell_height = final_heights[i]
        is_coast = False
        
        for neighbor in final_graph.cell_neighbors[i]:
            if neighbor < len(final_heights):
                neighbor_height = final_heights[neighbor]
                if (cell_height >= 20 and neighbor_height < 20) or \
                   (cell_height < 20 and neighbor_height >= 20):
                    is_coast = True
                    break
        
        if is_coast:
            coastline_count += 1
    
    print(f"Coastline cells: {coastline_count}")
    
    # Success criteria
    print("\n✅ SUCCESS CRITERIA:")
    print("-" * 40)
    
    fmg_min, fmg_max = 2, 51
    our_min, our_max = int(np.min(final_heights)), int(np.max(final_heights))
    
    print(f"Height range match: {our_min}-{our_max} vs FMG {fmg_min}-{fmg_max}")
    print(f"Max height gap: {abs(our_max - fmg_max)} (target: 0)")
    print(f"Min height gap: {abs(our_min - fmg_min)} (target: <5)")
    
    if our_max == fmg_max:
        print("✅ Maximum height matches perfectly!")
    
    if abs(our_min - fmg_min) < 5:
        print("✅ Minimum height is very close!")
    
    # Check cell count
    fmg_cells = 4264
    cell_count_diff = abs(total_cells - fmg_cells)
    print(f"\nCell count difference: {cell_count_diff} ({total_cells} vs {fmg_cells})")
    
    if cell_count_diff / fmg_cells < 0.05:  # Within 5%
        print("✅ Cell count is within 5% of FMG!")


if __name__ == "__main__":
    test_regraph_distribution()