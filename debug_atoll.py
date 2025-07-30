#!/usr/bin/env python3
"""Debug atoll template execution."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph

def debug_atoll():
    """Debug the atoll template step by step."""
    
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
    
    # Let's manually execute the atoll template steps and visualize each
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Step 1: Hill 1 75-80 50-60 45-55
    generator.add_hill(1, "75-80", "50-60", "45-55")
    heights1 = generator.heights.copy()  # Use generator.heights instead
    
    # Step 2: Hill 1.5 30-50 25-75 30-70
    generator.add_hill(1.5, "30-50", "25-75", "30-70")
    heights2 = generator.heights.copy()
    
    # Step 3: Hill .5 30-50 25-35 30-70
    generator.add_hill(0.5, "30-50", "25-35", "30-70")
    heights3 = generator.heights.copy()
    
    # Step 4: Smooth 1 0 0 0
    generator.smooth(1, "0", "0", "0")
    heights4 = generator.heights.copy()
    
    # Step 5: Multiply 0.2 25-100 0 0
    generator.modify("25-100", multiply=0.2)
    heights5 = generator.heights.copy()
    
    # Step 6: Hill 0.5 10-20 50-55 48-52
    print(f"\nBefore final hill: center cell heights = {generator.heights[len(generator.heights)//2-10:len(generator.heights)//2+10]}")
    generator.add_hill(0.5, "10-20", "50-55", "48-52")
    heights6 = generator.heights.copy()
    print(f"After final hill: center cell heights = {generator.heights[len(generator.heights)//2-10:len(generator.heights)//2+10]}")
    
    # Plot each step
    titles = [
        "Step 1: Hill 1 75-80 50-60 45-55",
        "Step 2: Hill 1.5 30-50 25-75 30-70", 
        "Step 3: Hill .5 30-50 25-35 30-70",
        "Step 4: Smooth 1",
        "Step 5: Multiply 0.2 25-100",
        "Step 6: Hill 0.5 10-20 50-55 48-52"
    ]
    
    all_heights = [heights1, heights2, heights3, heights4, heights5, heights6]
    
    for i, (ax, heights, title) in enumerate(zip(axes, all_heights, titles)):
        scatter = ax.scatter(graph.points[:, 0], graph.points[:, 1], 
                           c=heights, cmap='terrain', s=20, 
                           vmin=0, vmax=100)
        
        # Mark water line at height=20
        water_cells = heights < 20
        land_cells = heights >= 20
        
        ax.set_title(f"{title}\nWater: {water_cells.sum()}, Land: {land_cells.sum()}")
        ax.set_aspect('equal')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
        
        # Add contour at height=20
        from scipy.interpolate import griddata
        xi = np.linspace(0, width, 100)
        yi = np.linspace(0, height, 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata(graph.points, heights, (xi, yi), method='linear')
        ax.contour(xi, yi, zi, levels=[20], colors='blue', linewidths=2)
    
    plt.tight_layout()
    plt.savefig("debug_atoll_steps.png", dpi=150)
    print(f"\nSaved visualization to debug_atoll_steps.png")
    
    # Print height statistics
    print(f"\nFinal height statistics:")
    print(f"Min height: {heights6.min()}")
    print(f"Max height: {heights6.max()}")
    print(f"Heights < 20 (water): {(heights6 < 20).sum()}")
    print(f"Heights >= 20 (land): {(heights6 >= 20).sum()}")
    print(f"Heights in range 10-20: {((heights6 >= 10) & (heights6 < 20)).sum()}")
    print(f"Heights exactly 20: {(heights6 == 20).sum()}")

if __name__ == "__main__":
    debug_atoll()