#!/usr/bin/env python3
"""Debug triangular artifacts in water areas."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig, get_boundary_points, get_jittered_grid
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph

def visualize_voronoi_structure():
    """Visualize the Voronoi diagram to debug triangular artifacts."""
    
    # Simple test case
    width = 200
    height = 150
    cells_desired = 100
    seed = "debug"
    
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    
    # Calculate spacing
    spacing = np.sqrt((width * height) / cells_desired)
    spacing = round(spacing, 2)
    
    # Get points separately
    grid_points = get_jittered_grid(width, height, spacing, seed)
    boundary_points = get_boundary_points(width, height, spacing)
    
    print(f"Grid points: {len(grid_points)}")
    print(f"Boundary points: {len(boundary_points)}")
    print(f"Boundary points sample: {boundary_points[:5]}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Show just grid points
    ax = axes[0, 0]
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=20, label='Grid points')
    ax.set_xlim(-20, width + 20)
    ax.set_ylim(-20, height + 20)
    ax.set_title('Grid Points Only')
    ax.set_aspect('equal')
    ax.legend()
    
    # 2. Show grid + boundary points
    ax = axes[0, 1]
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=20, label='Grid points')
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=40, marker='x', label='Boundary points')
    ax.set_xlim(-20, width + 20)
    ax.set_ylim(-20, height + 20)
    ax.set_title('Grid + Boundary Points')
    ax.set_aspect('equal')
    ax.legend()
    
    # 3. Voronoi diagram without boundary points
    ax = axes[1, 0]
    vor_no_boundary = Voronoi(grid_points)
    voronoi_plot_2d(vor_no_boundary, ax=ax, show_vertices=False, line_colors='blue', line_width=1, point_size=5)
    ax.set_xlim(-20, width + 20)
    ax.set_ylim(-20, height + 20)
    ax.set_title('Voronoi WITHOUT Boundary Points')
    ax.set_aspect('equal')
    
    # 4. Voronoi diagram with boundary points
    ax = axes[1, 1]
    all_points = np.vstack([grid_points, boundary_points])
    vor_with_boundary = Voronoi(all_points)
    voronoi_plot_2d(vor_with_boundary, ax=ax, show_vertices=False, line_colors='green', line_width=1, point_size=5)
    ax.set_xlim(-20, width + 20)
    ax.set_ylim(-20, height + 20)
    ax.set_title('Voronoi WITH Boundary Points')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = Path("debug_voronoi_structure.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved debug visualization to {output_path}")
    
    # Now test with actual heightmap
    print("\n=== Testing with heightmap ===")
    
    # Generate full graph
    graph = generate_voronoi_graph(config, seed)
    
    # Create simple heightmap (mostly water)
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    # Create a simple island in the center
    graph.heights = np.zeros(len(graph.points), dtype=np.uint8)
    cx, cy = width / 2, height / 2
    for i, (x, y) in enumerate(graph.points):
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < 30:
            graph.heights[i] = 30
    
    # Run features markup
    features = Features(graph)
    features.markup_grid()
    
    # Run regraph
    packed = regraph(graph)
    
    print(f"Original cells: {len(graph.points)}")
    print(f"Packed cells: {len(packed.points)}")
    
    # Create another figure showing the heightmap
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Plot packed cells colored by height
    scatter = ax.scatter(packed.points[:, 0], packed.points[:, 1], 
                        c=packed.heights, cmap='terrain', s=50, 
                        vmin=0, vmax=100, edgecolors='black', linewidth=0.5)
    
    # Add water line
    water_mask = packed.heights < 20
    ax.scatter(packed.points[water_mask, 0], packed.points[water_mask, 1], 
              c='blue', s=30, alpha=0.5, label='Water cells')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title('Packed Cells with Heights')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Height')
    ax.legend()
    
    output_path2 = Path("debug_packed_cells.png")
    plt.savefig(output_path2, dpi=150)
    print(f"Saved packed cells visualization to {output_path2}")

if __name__ == "__main__":
    visualize_voronoi_structure()