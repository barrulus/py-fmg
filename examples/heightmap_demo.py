#!/usr/bin/env python3
"""
Demo script showing heightmap generation capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template, list_templates


def visualize_heightmap(heights: np.ndarray, title: str, width: int, height: int):
    """Visualize a heightmap as a heatmap."""
    # Reshape to 2D grid
    n_cells = len(heights)
    cells_x = int(np.sqrt(n_cells * width / height))
    cells_y = int(np.sqrt(n_cells * height / width))
    
    # Create a 2D array (might not be perfect rectangle)
    grid = np.zeros((cells_y, cells_x))
    for i in range(min(n_cells, cells_x * cells_y)):
        y = i // cells_x
        x = i % cells_x
        if y < cells_y:
            grid[y, x] = heights[i]
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='terrain', vmin=0, vmax=100)
    plt.colorbar(label='Height')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')


def main():
    """Demonstrate heightmap generation."""
    print("Py-FMG Heightmap Generation Demo")
    print("=" * 40)
    
    # Set up configuration
    width, height = 300, 300
    cells_desired = 2500
    
    print(f"\nGenerating Voronoi graph ({cells_desired} cells)...")
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed="demo123")
    print(f"Generated {len(graph.points)} cells")
    
    # Create heightmap generator
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing
    )
    
    # Generate different heightmap types
    templates_to_demo = ['volcano', 'archipelago', 'continents', 'highIsland']
    
    plt.figure(figsize=(16, 16))
    
    for i, template_name in enumerate(templates_to_demo, 1):
        print(f"\nGenerating {template_name} heightmap...")
        
        generator = HeightmapGenerator(heightmap_config, graph)
        template = get_template(template_name)
        heights = generator.from_template(template, seed=f"{template_name}_demo")
        
        # Calculate statistics
        land_pct = np.sum(heights >= 20) / len(heights) * 100
        avg_height = np.mean(heights)
        max_height = np.max(heights)
        
        print(f"  Land: {land_pct:.1f}%")
        print(f"  Average height: {avg_height:.1f}")
        print(f"  Max height: {max_height}")
        
        # Visualize
        plt.subplot(2, 2, i)
        
        # Reshape for visualization
        cells_x = graph.cells_x
        cells_y = graph.cells_y
        grid = np.zeros((cells_y, cells_x))
        
        for idx in range(min(len(heights), cells_x * cells_y)):
            y = idx // cells_x
            x = idx % cells_x
            if y < cells_y:
                grid[y, x] = heights[idx]
        
        plt.imshow(grid, cmap='terrain', vmin=0, vmax=100)
        plt.colorbar(label='Height')
        plt.title(f'{template_name.capitalize()} ({land_pct:.1f}% land)')
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('heightmap_examples.png', dpi=150)
    print("\nSaved visualization to heightmap_examples.png")
    
    # List all available templates
    print("\nAvailable heightmap templates:")
    for template in list_templates():
        print(f"  - {template}")


if __name__ == "__main__":
    main()