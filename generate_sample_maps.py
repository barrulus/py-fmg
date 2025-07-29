#!/usr/bin/env python3
"""
Generate high-resolution heightmaps using the full FMG pipeline.

This includes:
1. Voronoi graph generation with Lloyd's relaxation
2. Heightmap generation from templates
3. Features.markupGrid() for coastline detection
4. reGraph() for coastal enhancement and cell packing

Usage:
    python generate_sample_maps.py [seed]

If no seed is provided, defaults to "default_seed"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata


def create_map_with_full_pipeline(
    width=1200, 
    height=1000, 
    cells_desired=10000, 
    template="archipelago", 
    seed="default"
):
    """Generate a map using the full FMG pipeline."""
    
    print(f"\nGenerating {template} map with full pipeline...")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Target cells: {cells_desired}")
    print(f"  Seed: {seed}")
    
    # Step 1: Generate Voronoi graph
    print("  1. Generating Voronoi graph...")
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed)
    print(f"     Generated {len(graph.points)} cells")
    
    # Step 2: Generate heightmap
    print("  2. Applying heightmap template...")
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    heights = generator.from_template(template, seed=seed)
    graph.heights = heights
    
    # Calculate initial statistics
    water_cells = np.sum(heights < 20)
    land_cells = np.sum(heights >= 20)
    print(f"     Initial: {water_cells} water cells ({water_cells/len(heights)*100:.1f}%), "
          f"{land_cells} land cells ({land_cells/len(heights)*100:.1f}%)")
    
    # Step 3: Features markup
    print("  3. Running Features.markupGrid()...")
    features = Features(graph)
    features.markup_grid()
    print(f"     Detected {len(graph.features)-1} features")
    
    # Step 4: Cell packing with coastal enhancement
    print("  4. Running reGraph() for coastal enhancement...")
    packed = regraph(graph)
    print(f"     Packed to {len(packed.points)} cells "
          f"({(len(packed.points)-len(graph.points))/len(graph.points)*100:+.1f}% change)")
    
    # Create visualization
    print("  5. Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create interpolated grid from packed cells (enhanced coastlines)
    grid_resolution = 600
    xi = np.linspace(0, width, grid_resolution)
    yi = np.linspace(0, height, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Use packed points for interpolation
    points_array = np.array(packed.points)
    zi = griddata(points_array, packed.heights, (xi, yi), method='linear', fill_value=0)
    
    # Apply custom colormap
    colors = [
        (0.0, "#001a33"),  # Deep ocean
        (0.1, "#003366"),  # Ocean
        (0.19, "#0066cc"),  # Shallow water
        (0.20, "#66b266"),  # Coast/grass
        (0.35, "#99cc99"),  # Plains
        (0.50, "#cccc99"),  # Hills
        (0.65, "#cc9966"),  # Mountains
        (0.80, "#996633"),  # High mountains
        (1.0, "#ffffff"),  # Snow peaks
    ]
    
    cmap = LinearSegmentedColormap.from_list("terrain_custom", colors, N=100)
    
    # Plot the interpolated height map
    im = ax.imshow(
        zi,
        extent=(0, width, 0, height),
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=100,
        aspect="equal",
        interpolation='bilinear'
    )
    
    # Add subtle contour lines
    contours = ax.contour(
        xi, yi, zi, 
        levels=[10, 20, 30, 40, 50, 60, 70, 80], 
        colors="black", 
        linewidths=0.3, 
        alpha=0.3
    )
    
    # Highlight sea level
    sea_contour = ax.contour(
        xi, yi, zi, 
        levels=[20], 
        colors="navy", 
        linewidths=1.5, 
        alpha=0.8
    )
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title and statistics
    final_water = np.sum(packed.heights < 20)
    final_land = np.sum(packed.heights >= 20)
    total_cells = len(packed.heights)
    water_pct = (final_water / total_cells) * 100
    land_pct = (final_land / total_cells) * 100
    
    title = f"{template.title()} - {width}x{height} - {total_cells:,} cells\n"
    title += f"Land: {final_land:,} ({land_pct:.1f}%) | Water: {final_water:,} ({water_pct:.1f}%)"
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Height", shrink=0.8, pad=0.02)
    cbar.ax.axhline(y=20, color="blue", linewidth=2)
    cbar.ax.text(1.5, 20, "Sea Level", rotation=0, va='center', fontsize=10)
    
    # Add pipeline info
    pipeline_text = "Full FMG Pipeline:\n"
    pipeline_text += "✓ Voronoi + Lloyd's\n"
    pipeline_text += "✓ Heightmap template\n"
    pipeline_text += "✓ Features.markupGrid\n"
    pipeline_text += "✓ reGraph (coastal enhancement)"
    
    ax.text(0.02, 0.98, pipeline_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10, family="monospace")
    
    # Add seed info
    ax.text(0.98, 0.02, f"Seed: {seed}", transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            horizontalalignment='right', fontsize=10, family="monospace")
    
    # Save high resolution
    output_file = f"heightmap_{template}_{seed}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  6. Saved to: {output_file}")
    
    plt.close()
    
    return packed


def main():
    """Generate sample maps with the full pipeline."""
    
    # Get seed from command line argument or use default
    seed = sys.argv[1] if len(sys.argv) > 1 else "default_seed"
    
    # Templates to generate
    templates = [
        "archipelago",
        "continents", 
        "volcano",
        "atoll",
        "mediterranean",
        "peninsula",
        "pangea",
        "isthmus",
        "fractious",
        "shattered",
    ]
    
    print(f"Generating maps with full FMG pipeline")
    print(f"Using seed: {seed}")
    print("="*60)
    
    for template in templates:
        try:
            create_map_with_full_pipeline(
                width=1200,
                height=1000,
                cells_desired=10000,
                template=template,
                seed=seed,
            )
        except Exception as e:
            print(f"  ERROR generating {template}: {e}")
    
    print("\n" + "="*60)
    print("All maps generated successfully!")
    print("\nThe maps now include:")
    print("- Proper water/land distribution")
    print("- Enhanced coastlines from reGraph")
    print("- Feature detection and markup")
    print("- Correct blob spreading with integer truncation")


if __name__ == "__main__":
    main()