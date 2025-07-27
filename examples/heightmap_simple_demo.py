#!/usr/bin/env python3
"""
Simple demo script showing heightmap generation capabilities.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template, list_templates


def main():
    """Demonstrate heightmap generation."""
    print("Py-FMG Heightmap Generation Demo")
    print("=" * 40)
    
    # Set up configuration
    width, height = 100, 100
    cells_desired = 100
    
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
    templates_to_demo = ['volcano', 'archipelago', 'continents', 'lowIsland']
    
    for template_name in templates_to_demo:
        print(f"\n{template_name.upper()} Template:")
        print("-" * 30)
        
        generator = HeightmapGenerator(heightmap_config, graph)
        template = get_template(template_name)
        heights = generator.from_template(template, seed=f"{template_name}_demo")
        
        # Calculate statistics
        land_cells = np.sum(heights >= 20)
        water_cells = np.sum(heights < 20)
        land_pct = land_cells / len(heights) * 100
        avg_height = np.mean(heights)
        max_height = np.max(heights)
        min_height = np.min(heights)
        
        print(f"  Total cells: {len(heights)}")
        print(f"  Land cells: {land_cells} ({land_pct:.1f}%)")
        print(f"  Water cells: {water_cells} ({100-land_pct:.1f}%)")
        print(f"  Height range: {min_height}-{max_height}")
        print(f"  Average height: {avg_height:.1f}")
        
        # Show height distribution
        bins = [0, 10, 20, 30, 50, 70, 90, 100]
        hist, _ = np.histogram(heights, bins=bins)
        print("  Height distribution:")
        for i in range(len(bins)-1):
            bar = '#' * int(hist[i] / max(hist) * 20)
            print(f"    {bins[i]:3d}-{bins[i+1]:3d}: {bar} ({hist[i]})")
    
    # List all available templates
    print("\n\nAll available heightmap templates:")
    print("-" * 30)
    for template in list_templates():
        print(f"  - {template}")
    
    # Custom heightmap example
    print("\n\nCustom Heightmap Example:")
    print("-" * 30)
    print("Creating a custom island with central mountain...")
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Custom island generation
    generator.add_hill(1, 80, "45-55", "45-55")  # Central peak
    generator.add_hill(3, 30, "30-70", "30-70")  # Surrounding hills
    generator.smooth(2)  # Smooth terrain
    generator.mask(2)  # Fade edges to create island
    generator.add_pit(2, 15, "20-30", "60-80")  # Add some valleys
    
    heights = generator.heights
    
    # Stats for custom heightmap
    land_pct = np.sum(heights >= 20) / len(heights) * 100
    print(f"  Land percentage: {land_pct:.1f}%")
    print(f"  Max elevation: {np.max(heights)}")
    print(f"  Average elevation: {np.mean(heights):.1f}")


if __name__ == "__main__":
    main()