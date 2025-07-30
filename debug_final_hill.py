#!/usr/bin/env python3
"""Debug why the final hill in atoll isn't being added."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def debug_final_hill():
    """Debug the final hill placement."""
    
    # Use same parameters as the actual test
    width = 1200
    height = 1000
    cells_desired = 10000
    seed = "654321"  # map seed from test
    
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed="123456")  # grid seed from test
    
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Run the full atoll template
    heights_initial = generator.from_template("atoll", seed=seed)
    
    # Now let's manually test adding a hill at the center
    print(f"Map dimensions: {width}x{height}")
    print(f"50-55% of width: {0.5*width}-{0.55*width} = 600-660")
    print(f"48-52% of height: {0.48*height}-{0.52*height} = 480-520")
    
    # Find cells in the target region
    target_x_min, target_x_max = 0.5 * width, 0.55 * width
    target_y_min, target_y_max = 0.48 * height, 0.52 * height
    
    cells_in_region = []
    for i, (x, y) in enumerate(graph.points):
        if target_x_min <= x <= target_x_max and target_y_min <= y <= target_y_max:
            cells_in_region.append(i)
    
    print(f"\nFound {len(cells_in_region)} cells in the target region")
    
    if len(cells_in_region) > 0:
        print(f"Sample cells in region:")
        for i in cells_in_region[:5]:
            print(f"  Cell {i} at ({graph.points[i][0]:.1f}, {graph.points[i][1]:.1f}), height: {heights_initial[i]}")
    
    # Let's also check the atoll template result more broadly
    center_x, center_y = width / 2, height / 2
    distances = np.sqrt((graph.points[:, 0] - center_x)**2 + (graph.points[:, 1] - center_y)**2)
    
    # Check different radius bands
    for r in [50, 100, 150, 200]:
        mask = distances < r
        water = np.sum(heights_initial[mask] < 20)
        land = np.sum(heights_initial[mask] >= 20)
        avg_height = np.mean(heights_initial[mask])
        print(f"\nWithin {r} units of center: {water} water, {land} land (avg height: {avg_height:.1f})")
    
    # Find the cell closest to exact center
    center_cell = np.argmin(distances)
    print(f"\nClosest cell to center:")
    print(f"  Index: {center_cell}")
    print(f"  Position: {graph.points[center_cell]}")
    print(f"  Height: {heights_initial[center_cell]}")
    print(f"  Is land: {heights_initial[center_cell] >= 20}")
    
    # Check if there's a systematic issue with coordinate ranges
    print(f"\nChecking coordinate interpretation:")
    print(f"  Width: {width}, Height: {height}")
    print(f"  Generator width: {generator.config.width}, height: {generator.config.height}")
    print(f"  Graph dimensions: {graph.graph_width}x{graph.graph_height}")

if __name__ == "__main__":
    debug_final_hill()