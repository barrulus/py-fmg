"""End-to-end test for the complete map generation pipeline."""

import sys
import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.config.heightmap_templates import TEMPLATES

# Test configuration constants
TEST_WIDTH = 1200
TEST_HEIGHT = 1000
TEST_CELLS_DESIRED = 10000
DEFAULT_GRID_SEED = "123456"
DEFAULT_MAP_SEED = "654321"


def test_full_pipeline_with_visualization(template="atoll", grid_seed=None, map_seed=None):
    """Test the complete map generation pipeline from seed to packed graph.
    
    Args:
        template: Heightmap template name
        grid_seed: Seed for Voronoi generation (defaults to DEFAULT_GRID_SEED)
        map_seed: Seed for heightmap generation (defaults to DEFAULT_MAP_SEED)
    """
    # Use defaults if not provided
    grid_seed = grid_seed or DEFAULT_GRID_SEED
    map_seed = map_seed or DEFAULT_MAP_SEED
    
    # Stage 1: Configure grid parameters
    width = TEST_WIDTH
    height = TEST_HEIGHT
    cells_desired = TEST_CELLS_DESIRED

    config = GridConfig(width=width, height=height, cells_desired=cells_desired)

    # Stage 2: Generate Voronoi graph
    print("Generating Voronoi graph...")
    voronoi_graph = generate_voronoi_graph(config, grid_seed)
    assert voronoi_graph is not None
    assert len(voronoi_graph.points) > 0
    assert voronoi_graph.cells_x > 0
    assert voronoi_graph.cells_y > 0
    print(f"Generated {len(voronoi_graph.points)} Voronoi cells")

    # Stage 3: Generate heightmap
    print("Generating heightmap...")
    heightmap_config = HeightmapConfig(
        width=int(width),
        height=int(height),
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=cells_desired,
        spacing=voronoi_graph.spacing,
    )

    heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = heightmap_gen.from_template(template, map_seed)

    # Verify heightmap properties
    assert heights is not None
    # Heightmap generator returns per-cell heights (1D array)
    assert len(heights) == len(voronoi_graph.points)
    assert heights.dtype == np.uint8
    assert np.any(heights > 20)  # Should have some land
    assert np.any(heights <= 20)  # Should have some water

    # Assign heights to graph
    voronoi_graph.heights = heights
    print(f"Generated heightmap with shape {heights.shape}")
    print(f"Height range: {heights.min()} to {heights.max()}")
    print(f"Percentage of land: {(heights > 20).sum() / heights.size * 100:.1f}%")

    # Stage 4: Mark up coastlines with Features
    print("Marking up coastlines...")
    features = Features(voronoi_graph)
    features.markup_grid()

    # Verify that distance_field was created
    assert hasattr(voronoi_graph, "distance_field")
    assert voronoi_graph.distance_field is not None
    print("Coastline markup completed")

    # Stage 5: Perform reGraph coastal resampling
    print("Performing reGraph coastal resampling...")
    packed_graph = regraph(voronoi_graph)

    # Verify regraph results
    assert packed_graph is not None
    assert packed_graph.points is not None
    assert packed_graph.heights is not None
    assert len(packed_graph.points) > 0

    packed_heights = packed_graph.heights

    print(f"Packed graph has {len(packed_graph.points)} cells")
    print(f"Cell reduction: {len(voronoi_graph.points)} -> {len(packed_graph.points)}")

    # Verify reasonable cell count
    # Note: archipelago template may increase cells due to many coastlines
    # For templates with many islands, packed graph may have more cells due to coastal enhancement
    assert len(packed_graph.points) > 100  # Should still have a reasonable number
    # The actual reduction/increase depends on the template and water/land distribution

    # Stage 6: Generate visualization matching generate_sample_maps.py
    print("Generating visualization...")
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)

    # Import needed for interpolation
    from scipy.interpolate import griddata
    from matplotlib.colors import LinearSegmentedColormap

    # Create a single plot like generate_sample_maps.py
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create interpolated grid from packed cells (enhanced coastlines)
    grid_resolution = 600  # Lower resolution for test
    xi = np.linspace(0, width, grid_resolution)
    yi = np.linspace(0, height, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Use packed points for interpolation
    points_array = np.array(packed_graph.points)
    zi = griddata(points_array, packed_heights, (xi, yi), method="linear", fill_value=0)

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
        interpolation="bilinear",
    )

    # Add subtle contour lines
    contours = ax.contour(
        xi,
        yi,
        zi,
        levels=[10, 20, 30, 40, 50, 60, 70, 80],
        colors="black",
        linewidths=0.3,
        alpha=0.3,
    )

    # Highlight sea level
    sea_contour = ax.contour(
        xi, yi, zi, levels=[20], colors="navy", linewidths=1.5, alpha=0.8
    )

    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title and statistics
    final_water = np.sum(packed_heights < 20)
    final_land = np.sum(packed_heights >= 20)
    total_cells = len(packed_heights)
    water_pct = (final_water / total_cells) * 100
    land_pct = (final_land / total_cells) * 100

    title = f"{template.title()} - {width}x{height} - {total_cells:,} cells\n"
    title += f"Land: {final_land:,} ({land_pct:.1f}%) | Water: {final_water:,} ({water_pct:.1f}%)"
    ax.set_title(title, fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Height", shrink=0.8, pad=0.02)
    cbar.ax.axhline(y=20, color="blue", linewidth=2)
    cbar.ax.text(1.5, 20, "Sea Level", rotation=0, va="center", fontsize=10)

    # Add pipeline info
    pipeline_text = "Full FMG Pipeline:\n"
    pipeline_text += "✓ Voronoi + Lloyd's\n"
    pipeline_text += "✓ Heightmap template\n"
    pipeline_text += "✓ Features.markupGrid\n"
    pipeline_text += "✓ reGraph (coastal enhancement)"

    ax.text(
        0.02,
        0.98,
        pipeline_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=10,
        family="monospace",
    )

    # Add seed info
    ax.text(
        0.98,
        0.02,
        f"Seeds: grid={grid_seed}, map={map_seed}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        horizontalalignment="right",
        fontsize=10,
        family="monospace",
    )

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"heightmap_{template}_{grid_seed}+{map_seed}_{timestamp}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Visualization saved to {output_path}")
    print("\nPipeline test completed successfully!")

    # Return results for potential further testing
    return {
        "voronoi_graph": voronoi_graph,
        "heights": heights,
        "packed_graph": packed_graph,
        "packed_heights": packed_heights,
        "features": features,
        "visualization_path": output_path,
    }


def test_pipeline_error_handling():
    """Test that the pipeline properly handles missing Features.markup_grid() call."""

    # Generate basic components
    config = GridConfig(width=TEST_WIDTH, height=TEST_HEIGHT, cells_desired=TEST_CELLS_DESIRED)
    voronoi_graph = generate_voronoi_graph(config, "error-test")

    heightmap_config = HeightmapConfig(
        width=TEST_WIDTH,
        height=TEST_HEIGHT,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=TEST_CELLS_DESIRED,
        spacing=voronoi_graph.spacing,
    )

    heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = heightmap_gen.from_template("atoll", "error-test")
    voronoi_graph.heights = heights

    # Try to call regraph WITHOUT Features.markup_grid()
    with pytest.raises(ValueError, match="graph.distance_field not found"):
        regraph(voronoi_graph)

    print(
        "Error handling test passed: regraph correctly requires Features.markup_grid()"
    )


def main():
    """Run end-to-end tests with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FMG pipeline with various templates")
    parser.add_argument("--template", default="all", 
                       help="Template name or 'all' to test all templates")
    parser.add_argument("--grid-seed", default=DEFAULT_GRID_SEED,
                       help=f"Grid seed (default: {DEFAULT_GRID_SEED})")
    parser.add_argument("--map-seed", default=DEFAULT_MAP_SEED,
                       help=f"Map seed (default: {DEFAULT_MAP_SEED})")
    
    args = parser.parse_args()
    
    # Run error handling test first
    print("Running error handling test...")
    test_pipeline_error_handling()
    print()
    
    # Determine which templates to test
    if args.template == "all":
        templates_to_test = list(TEMPLATES.keys())
        print(f"Testing all {len(templates_to_test)} templates...")
    else:
        templates_to_test = [args.template]
        print(f"Testing template: {args.template}")
    
    print(f"Grid seed: {args.grid_seed}")
    print(f"Map seed: {args.map_seed}")
    print("=" * 60)
    
    # Run tests for each template
    successful = 0
    failed = []
    
    for template in templates_to_test:
        try:
            print(f"\nTesting {template} template...")
            results = test_full_pipeline_with_visualization(
                template=template,
                grid_seed=args.grid_seed,
                map_seed=args.map_seed
            )
            successful += 1
        except Exception as e:
            print(f"ERROR: Template {template} failed: {e}")
            failed.append((template, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Summary: {successful}/{len(templates_to_test)} templates succeeded")
    if failed:
        print("\nFailed templates:")
        for template, error in failed:
            print(f"  - {template}: {error}")
    else:
        print("\nAll templates passed successfully!")


if __name__ == "__main__":
    main()
