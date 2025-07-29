#!/usr/bin/env python3
"""
Visualize the Voronoi diagram with heightmap colors.
Generates an image showing the height values as a color map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed


def create_voronoi_patches(graph, heights):
    """Create matplotlib patches for each Voronoi cell."""
    patches = []

    for i in range(len(graph.points)):
        # Get the vertices for this cell
        vertex_indices = graph.cell_vertices[i]
        if not vertex_indices:
            continue

        # Get vertex coordinates
        vertices = []
        for v_idx in vertex_indices:
            if 0 <= v_idx < len(graph.vertex_coordinates):
                vertices.append(graph.vertex_coordinates[v_idx])

        if len(vertices) >= 3:  # Need at least 3 vertices for a polygon
            # Create a polygon patch
            polygon = Polygon(vertices, closed=True)
            patches.append(polygon)

    return patches


def visualize_heightmap(
    width=300, height=300, cells_desired=10000, template="archipelago", seed="123456"
):
    """
    Generate and visualize a heightmap on a Voronoi diagram.

    Args:
        width: Map width
        height: Map height
        cells_desired: Number of Voronoi cells
        template: Heightmap template name
        seed: Random seed
    """
    print(f"Generating Voronoi graph with {cells_desired} cells...")

    # Generate Voronoi graph
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed)

    print(f"Generated {len(graph.points)} cells")

    # Generate heightmap
    print(f"Applying '{template}' heightmap template...")
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

    # Statistics
    print(f"\nHeightmap statistics:")
    print(f"  Min height: {np.min(heights)}")
    print(f"  Max height: {np.max(heights)}")
    print(f"  Mean height: {np.mean(heights):.1f}")
    print(
        f"  Land cells (h>=20): {np.sum(heights >= 20)} ({np.sum(heights >= 20)/len(heights)*100:.1f}%)"
    )
    print(
        f"  Water cells (h<20): {np.sum(heights < 20)} ({np.sum(heights < 20)/len(heights)*100:.1f}%)"
    )

    # Create visualization
    print("\nCreating visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Voronoi cells colored by height
    patches = create_voronoi_patches(graph, heights)
    if patches:
        # Create a collection with height-based colors
        p = PatchCollection(patches, cmap="terrain", edgecolor="none")
        p.set_array(heights[: len(patches)])
        p.set_clim(0, 100)
        ax1.add_collection(p)

        # Add colorbar
        cbar1 = plt.colorbar(p, ax=ax1, label="Height")
        cbar1.ax.axhline(y=20, color="blue", linewidth=2, label="Sea level")

    # Set limits and labels for left plot
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect("equal")
    ax1.set_title(f"Voronoi Heightmap - {template} template\n{len(graph.points)} cells")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Right plot: Height distribution as image
    # Create a grid for interpolation
    grid_resolution = 200
    xi = np.linspace(0, width, grid_resolution)
    yi = np.linspace(0, height, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Simple nearest-neighbor interpolation
    from scipy.interpolate import griddata

    points_array = np.array(graph.points)
    zi = griddata(points_array, heights, (xi, yi), method="nearest")

    # Plot as image
    im = ax2.imshow(
        zi,
        extent=(0, width, 0, height),
        origin="lower",
        cmap="terrain",
        vmin=0,
        vmax=100,
        aspect="equal",
    )

    # Add contour lines
    contour_levels = [20, 40, 60, 80]  # Sea level at 20
    contours = ax2.contour(
        xi, yi, zi, levels=contour_levels, colors="black", linewidths=0.5, alpha=0.5
    )
    ax2.clabel(contours, inline=True, fontsize=8)

    # Add sea level contour in blue
    sea_contour = ax2.contour(
        xi, yi, zi, levels=[20], colors="blue", linewidths=2, alpha=0.8
    )

    # Add colorbar
    cbar2 = plt.colorbar(im, ax=ax2, label="Height")

    # Set labels for right plot
    ax2.set_title(f"Interpolated Height Map\nSea level = 20 (blue line)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Overall title
    fig.suptitle(f"Heightmap Visualization - Seed: {seed}", fontsize=16)

    plt.tight_layout()

    # Save the figure in high resolution
    output_file = f"heightmap_{template}_{seed}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nHigh-res visualization saved to: {output_file}")

    plt.show()


def create_simple_height_image(
    width=600, height=600, cells_desired=5000, template="archipelago", seed="visualize"
):
    """Create a simple height map image without Voronoi cell borders."""

    print(f"\nCreating simple height map image...")

    # Generate Voronoi graph
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed)

    # Generate heightmap
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
    
    # Calculate land/water statistics
    total_cells = len(heights)
    land_cells = np.sum(heights >= 20)
    water_cells = np.sum(heights < 20)
    land_pct = (land_cells / total_cells) * 100
    water_pct = (water_cells / total_cells) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create interpolated grid
    grid_resolution = 300
    xi = np.linspace(0, width, grid_resolution)
    yi = np.linspace(0, height, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate heights
    from scipy.interpolate import griddata

    points_array = np.array(graph.points)
    zi = griddata(points_array, heights, (xi, yi), method="cubic", fill_value=0)

    # Apply custom colormap for terrain
    # Water (0-19) -> shades of blue
    # Land (20-100) -> green to brown to white
    from matplotlib.colors import LinearSegmentedColormap

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

    n_bins = 100
    cmap_name = "terrain_custom"
    cm_custom = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Plot the interpolated height map
    im = ax.imshow(
        zi,
        extent=(0, width, 0, height),
        origin="lower",
        cmap=cm_custom,
        vmin=0,
        vmax=100,
        aspect="equal",
    )

    # Add subtle contour lines
    ax.contour(
        xi, yi, zi, levels=10, colors="black", linewidths=0.3, alpha=0.3
    )

    # Highlight sea level
    ax.contour(
        xi, yi, zi, levels=[20], colors="navy", linewidths=1.5, alpha=0.8
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Height", shrink=0.8)
    cbar.ax.axhline(y=20, color="blue", linewidth=2, label="Sea level")

    # Keep axes visible with labels
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    
    # Create title with land/water statistics
    title_lines = [
        f"{template.title()} - {width}x{height} - {total_cells:,} cells",
        f"Land: {land_cells:,} cells ({land_pct:.1f}%) | Water: {water_cells:,} cells ({water_pct:.1f}%)"
    ]
    ax.set_title("\n".join(title_lines), fontsize=14, pad=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Add statistics text box
    stats_text = f"Total Cells: {total_cells:,}\nLand: {land_cells:,} ({land_pct:.1f}%)\nWater: {water_cells:,} ({water_pct:.1f}%)\nSeed: {seed}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            verticalalignment="top", fontsize=10, family="monospace")

    # Save the image in high resolution
    output_file = f"heightmap_{template}_{seed}.png"
    plt.savefig(output_file, dpi=600, bbox_inches="tight", pad_inches=0.1)
    print(f"High-res (600 DPI) height map saved to: {output_file}")

    plt.show()


def main():
    """Main function to generate visualizations."""

    # Generate visualization for archipelago
    visualize_heightmap(
        width=600,
        height=600,
        cells_desired=5000,
        template="archipelago",
        seed="test123",
    )

    # Also create a simple version
    create_simple_height_image(
        width=600,
        height=600,
        cells_desired=5000,
        template="archipelago",
        seed="test123",
    )


if __name__ == "__main__":
    main()
