#!/usr/bin/env python3
"""Debug atoll template in detail."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def debug_atoll_detailed():
    """Debug the atoll template focusing on the center."""
    
    width = 400
    height = 400
    cells_desired = 1000
    seed = "atoll_debug"
    
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed)
    
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Find cells near the center
    center_x, center_y = width / 2, height / 2
    distances = np.sqrt((graph.points[:, 0] - center_x)**2 + (graph.points[:, 1] - center_y)**2)
    center_cells = np.where(distances < 30)[0]  # cells within 30 units of center
    
    print(f"Tracking {len(center_cells)} cells near the center")
    print(f"Center cell indices: {center_cells[:10]}...")
    
    # Execute template step by step
    print("\n=== Step 1: Hill 1 75-80 50-60 45-55 ===")
    generator.add_hill(1, "75-80", "50-60", "45-55")
    print(f"Center heights after step 1: {generator.heights[center_cells]}")
    
    print("\n=== Step 2: Hill 1.5 30-50 25-75 30-70 ===")
    generator.add_hill(1.5, "30-50", "25-75", "30-70")
    print(f"Center heights after step 2: {generator.heights[center_cells]}")
    
    print("\n=== Step 3: Hill .5 30-50 25-35 30-70 ===")
    generator.add_hill(0.5, "30-50", "25-35", "30-70")
    print(f"Center heights after step 3: {generator.heights[center_cells]}")
    
    print("\n=== Step 4: Smooth 1 ===")
    generator.smooth(1, "0", "0", "0")
    print(f"Center heights after step 4: {generator.heights[center_cells]}")
    
    print("\n=== Step 5: Multiply 0.2 25-100 ===")
    heights_before_multiply = generator.heights[center_cells].copy()
    generator.modify("25-100", multiply=0.2)
    heights_after_multiply = generator.heights[center_cells].copy()
    print(f"Center heights before multiply: {heights_before_multiply}")
    print(f"Center heights after multiply: {heights_after_multiply}")
    print(f"Heights changed by multiply: {np.sum(heights_before_multiply != heights_after_multiply)} cells")
    
    print("\n=== Step 6: Hill 0.5 10-20 50-55 48-52 ===")
    print("This should add a hill in range 50-55% x, 48-52% y")
    print(f"That's approximately x={0.5*width}-{0.55*width}, y={0.48*height}-{0.52*height}")
    print(f"Which is x=200-220, y=192-208")
    
    heights_before_final = generator.heights.copy()
    generator.add_hill(0.5, "10-20", "50-55", "48-52")
    heights_after_final = generator.heights.copy()
    
    # Find which cells were modified
    modified_cells = np.where(heights_before_final != heights_after_final)[0]
    print(f"\nModified {len(modified_cells)} cells in final hill")
    
    if len(modified_cells) > 0:
        print(f"Sample of modified cells:")
        for i in modified_cells[:10]:
            print(f"  Cell {i} at ({graph.points[i][0]:.1f}, {graph.points[i][1]:.1f}): "
                  f"{heights_before_final[i]:.1f} -> {heights_after_final[i]:.1f} "
                  f"(+{heights_after_final[i] - heights_before_final[i]:.1f})")
    
    # Check final results
    print(f"\n=== Final Results ===")
    print(f"Center heights after all steps: {generator.heights[center_cells]}")
    water_in_center = np.sum(generator.heights[center_cells] < 20)
    land_in_center = np.sum(generator.heights[center_cells] >= 20)
    print(f"Center cells: {water_in_center} water, {land_in_center} land")
    
    # Check the exact center
    closest_to_center = np.argmin(distances)
    print(f"\nClosest cell to center: index {closest_to_center}")
    print(f"Position: {graph.points[closest_to_center]}")
    print(f"Final height: {generator.heights[closest_to_center]}")
    print(f"Is water: {generator.heights[closest_to_center] < 20}")

if __name__ == "__main__":
    debug_atoll_detailed()