#!/usr/bin/env python3
"""Debug volcano template execution."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph

def debug_volcano():
    """Debug the volcano template step by step."""
    
    width = 600
    height = 500
    cells_desired = 2000
    seed = "volcano_debug"
    
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
    
    # Volcano template:
    # Hill 1 90-100 44-56 40-60
    # Multiply 0.8 50-100 0 0
    # Range 1.5 30-55 45-55 40-60
    # Smooth 3 0 0 0
    # Hill 1.5 35-45 25-30 20-75
    # Hill 1 35-55 75-80 25-75
    # Hill 0.5 20-25 10-15 20-25
    # Mask 3 0 0 0
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Step 1: Hill 1 90-100 44-56 40-60 - Main volcanic cone
    print("Step 1: Hill 1 90-100 44-56 40-60")
    print(f"  This should place a tall hill (90-100) at center (44-56% x, 40-60% y)")
    generator.add_hill(1, "90-100", "44-56", "40-60")
    heights1 = generator.heights.copy()
    
    # Step 2: Multiply 0.8 50-100 0 0
    print("\nStep 2: Multiply 0.8 50-100 0 0")
    print(f"  This multiplies heights 50-100 by 0.8")
    generator.modify("50-100", multiply=0.8)
    heights2 = generator.heights.copy()
    
    # Step 3: Range 1.5 30-55 45-55 40-60
    print("\nStep 3: Range 1.5 30-55 45-55 40-60")
    generator.add_range(1.5, "30-55", "45-55", "40-60")
    heights3 = generator.heights.copy()
    
    # Step 4: Smooth 3 0 0 0
    print("\nStep 4: Smooth 3")
    generator.smooth(3, "0", "0", "0")
    heights4 = generator.heights.copy()
    
    # Step 5: Hill 1.5 35-45 25-30 20-75
    print("\nStep 5: Hill 1.5 35-45 25-30 20-75")
    generator.add_hill(1.5, "35-45", "25-30", "20-75")
    heights5 = generator.heights.copy()
    
    # Step 6: Hill 1 35-55 75-80 25-75
    print("\nStep 6: Hill 1 35-55 75-80 25-75")
    generator.add_hill(1, "35-55", "75-80", "25-75")
    heights6 = generator.heights.copy()
    
    # Step 7: Hill 0.5 20-25 10-15 20-25
    print("\nStep 7: Hill 0.5 20-25 10-15 20-25")
    generator.add_hill(0.5, "20-25", "10-15", "20-25")
    heights7 = generator.heights.copy()
    
    # Step 8: Mask 3 0 0 0
    print("\nStep 8: Mask 3")
    generator.mask(3)
    heights8 = generator.heights.copy()
    
    # Plot each step
    titles = [
        "Step 1: Main cone (90-100)",
        "Step 2: Multiply tall peaks", 
        "Step 3: Add range",
        "Step 4: Smooth 3x",
        "Step 5: Side hill NW",
        "Step 6: Side hill SE",
        "Step 7: Small hill corner",
        "Step 8: Mask edges",
        "Final (with regraph)"
    ]
    
    all_heights = [heights1, heights2, heights3, heights4, heights5, heights6, heights7, heights8]
    
    # Create interpolated visualizations
    from scipy.interpolate import griddata
    xi = np.linspace(0, width, 200)
    yi = np.linspace(0, height, 200)
    xi, yi = np.meshgrid(xi, yi)
    
    for i, (ax, heights, title) in enumerate(zip(axes[:-1], all_heights, titles[:-1])):
        # Interpolate for smooth visualization
        zi = griddata(graph.points, heights, (xi, yi), method='linear', fill_value=0)
        
        im = ax.imshow(zi, extent=(0, width, 0, height), origin='lower', 
                      cmap='terrain', vmin=0, vmax=100, aspect='equal')
        
        # Add contour at sea level
        ax.contour(xi, yi, zi, levels=[20], colors='blue', linewidths=2)
        
        water_cells = heights < 20
        land_cells = heights >= 20
        
        ax.set_title(f"{title}\nWater: {water_cells.sum()}, Land: {land_cells.sum()}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Final step: run through features and regraph
    graph.heights = heights8
    features = Features(graph)
    features.markup_grid()
    packed = regraph(graph)
    
    # Plot final packed result
    ax = axes[-1]
    zi_packed = griddata(packed.points, packed.heights, (xi, yi), method='linear', fill_value=0)
    im = ax.imshow(zi_packed, extent=(0, width, 0, height), origin='lower', 
                  cmap='terrain', vmin=0, vmax=100, aspect='equal')
    ax.contour(xi, yi, zi_packed, levels=[20], colors='blue', linewidths=2)
    ax.set_title(titles[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig("debug_volcano_steps.png", dpi=150)
    print(f"\nSaved visualization to debug_volcano_steps.png")
    
    # Analyze the center region
    center_x, center_y = width / 2, height / 2
    distances = np.sqrt((graph.points[:, 0] - center_x)**2 + (graph.points[:, 1] - center_y)**2)
    center_cells = np.where(distances < 50)[0]
    
    print(f"\nCenter region analysis (within 50 units):")
    print(f"  Number of cells: {len(center_cells)}")
    print(f"  Heights: min={heights8[center_cells].min()}, max={heights8[center_cells].max()}, avg={heights8[center_cells].mean():.1f}")
    print(f"  Water cells: {np.sum(heights8[center_cells] < 20)}")
    print(f"  Land cells: {np.sum(heights8[center_cells] >= 20)}")

if __name__ == "__main__":
    debug_volcano()