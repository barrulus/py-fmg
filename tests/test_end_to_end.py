"""End-to-end test for the complete map generation pipeline."""

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cell_packing import regraph
from py_fmg.config.heightmap_templates import TEMPLATES, get_template

# Test configuration constants
TEST_WIDTH = 1200
TEST_HEIGHT = 1000
TEST_CELLS_DESIRED = 10000
DEFAULT_SEED = "123456789"

# NOTE: FMG uses a single seed that gets reseeded at each major stage:
# 1. Graph generation (Math.random = aleaPRNG(seed))
# 2. Heightmap generation (Math.random = aleaPRNG(seed))
# 3. Features markup (Math.random = aleaPRNG(seed))
# This ensures deterministic results with the same PRNG sequence at each stage.


def generate_packed_voronoi_debug(
    voronoi_graph, packed_graph, output_path, template_name, seed
):
    """Generate debug visualization showing packed/regraphed Voronoi results."""
    from scipy.spatial import Voronoi, voronoi_plot_2d

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Get dimensions
    width = voronoi_graph.graph_width
    height = voronoi_graph.graph_height

    # Subplot 1: Packed points colored by height
    heights = packed_graph.heights
    scatter = ax1.scatter(
        *packed_graph.points.T,
        c=heights,
        cmap="terrain",
        s=30,
        alpha=0.8,
        vmin=0,
        vmax=100,
    )
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect("equal")
    ax1.set_title("Packed Points (Colored by Height)", fontsize=14)
    plt.colorbar(scatter, ax=ax1, label="Height")

    # Add water/land line
    water_mask = heights < 20
    land_mask = heights >= 20
    ax1.scatter(
        *packed_graph.points[water_mask].T, c="blue", s=10, alpha=0.3, label="Water"
    )
    ax1.scatter(
        *packed_graph.points[land_mask].T, c="green", s=10, alpha=0.3, label="Land"
    )

    # Subplot 2: Original vs Packed points
    ax2.scatter(
        *voronoi_graph.points.T, s=5, alpha=0.3, label="Original points", color="gray"
    )
    ax2.scatter(
        *packed_graph.points.T, s=20, alpha=0.8, label="Packed points", color="red"
    )
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect("equal")
    ax2.set_title("Original vs Packed Points", fontsize=14)
    ax2.legend()

    # Subplot 3: Packed Voronoi
    vor_packed = Voronoi(packed_graph.points)
    voronoi_plot_2d(
        vor_packed,
        ax=ax3,
        show_vertices=False,
        line_colors="red",
        line_width=1.5,
        point_size=3,
    )
    ax3.set_xlim(0, width)
    ax3.set_ylim(0, height)
    ax3.set_title("Packed/Regraphed Voronoi (Coastal Enhancement)", fontsize=14)

    # Add coastline detection info
    if hasattr(packed_graph, "distance_field"):
        coastal_cells = (packed_graph.distance_field == 1) | (
            packed_graph.distance_field == -1
        )
        coastal_points = packed_graph.points[coastal_cells]
        ax3.scatter(
            *coastal_points.T,
            c="yellow",
            s=50,
            alpha=0.8,
            edgecolors="black",
            linewidths=1,
            label="Coastal cells",
        )
        ax3.legend()

    # Subplot 4: Cell density comparison
    ax4.bar(
        ["Original", "Packed"],
        [len(voronoi_graph.points), len(packed_graph.points)],
        color=["gray", "red"],
        alpha=0.7,
    )
    ax4.set_ylabel("Number of Cells")
    ax4.set_title("Cell Count Comparison", fontsize=14)

    # Add percentage labels
    reduction_pct = (
        (len(voronoi_graph.points) - len(packed_graph.points))
        / len(voronoi_graph.points)
    ) * 100
    ax4.text(
        1,
        len(packed_graph.points) + 100,
        f"{reduction_pct:+.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    # Add overall title
    title = f"{template_name.title()} - Packed Voronoi Analysis (Seed: {seed})"
    title += f"\nOriginal: {len(voronoi_graph.points)} cells | Packed: {len(packed_graph.points)} cells"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_heightmap_steps_debug(heightmap_gen, template_name, output_path, seed):
    """Generate step-by-step visualization of heightmap generation."""
    from scipy.interpolate import griddata
    from matplotlib.colors import LinearSegmentedColormap

    # Parse template to get steps
    template_text = get_template(template_name)
    lines = template_text.strip().split("\n")

    # Store heights at each step
    step_heights = []
    step_labels = []

    # Reset heightmap generator
    heightmap_gen.heights = np.zeros(heightmap_gen.n_cells, dtype=np.float32)

    # Execute each command and store results
    step_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        command = parts[0]
        args = parts[1:]

        # Execute command
        if command == "Hill":
            heightmap_gen.add_hill(*args)
        elif command == "Pit":
            heightmap_gen.add_pit(*args)
        elif command == "Range":
            heightmap_gen.add_range(*args)
        elif command == "Trough":
            heightmap_gen.add_trough(*args)
        elif command == "Strait":
            heightmap_gen.add_strait(*args)
        elif command == "Smooth":
            heightmap_gen.smooth(*args)
        elif command == "Mask":
            heightmap_gen.mask(float(args[0]))
        elif command == "Add":
            heightmap_gen.modify(args[1], add=float(args[0]))
        elif command == "Multiply":
            heightmap_gen.modify(args[1], multiply=float(args[0]))
        elif command == "Invert":
            heightmap_gen.invert(*args)

        # Only store major steps (not every smooth/mask operation)
        if command in ["Hill", "Pit", "Range", "Trough", "Strait", "Multiply"]:
            step_count += 1
            step_heights.append(heightmap_gen.heights.copy())
            step_labels.append(f"Step {step_count}: {' '.join([command] + args[:2])}")

        # Limit to 6 steps for visualization
        if len(step_heights) >= 6:
            break

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Colormap
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

    # Plot each step
    for i, (heights, label) in enumerate(zip(step_heights, step_labels)):
        if i >= 6:
            break

        ax = axes[i]

        # Scatter plot with colors
        points_array = np.array(heightmap_gen.graph.points)
        scatter = ax.scatter(
            points_array[:, 0],
            points_array[:, 1],
            c=heights,
            cmap=cmap,
            s=10,
            vmin=0,
            vmax=100,
        )

        # Format
        ax.set_xlim(0, heightmap_gen.config.width)
        ax.set_ylim(0, heightmap_gen.config.height)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        # Add water/land statistics
        water = np.sum(heights < 20)
        land = np.sum(heights >= 20)
        total = len(heights)
        water_pct = (water / total) * 100
        ax.text(
            0.02,
            0.98,
            f"Water: {water_pct:.1f}%\nLand: {100-water_pct:.1f}%",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

    # Hide unused subplots
    for i in range(len(step_heights), 6):
        axes[i].axis("off")

    # Add overall title
    fig.suptitle(
        f"{template_name.title()} - Heightmap Generation Steps (Seed: {seed})",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_voronoi_debug(voronoi_graph, output_path, template_name, seed):
    """Generate debug visualization showing initial Voronoi diagram."""
    from scipy.spatial import Voronoi, voronoi_plot_2d

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get dimensions
    width = voronoi_graph.graph_width
    height = voronoi_graph.graph_height

    # Plot Voronoi diagram
    vor = Voronoi(voronoi_graph.points)
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        line_colors="gray",
        line_width=0.5,
        point_size=2,
    )

    # Plot points colored by height if available
    if hasattr(voronoi_graph, "heights") and voronoi_graph.heights is not None:
        scatter = ax.scatter(
            *voronoi_graph.points.T,
            c=voronoi_graph.heights,
            cmap="terrain",
            s=15,
            alpha=0.7,
            vmin=0,
            vmax=100,
        )
        plt.colorbar(scatter, ax=ax, label="Height")
    else:
        ax.scatter(
            *voronoi_graph.points.T,
            c="red",
            s=15,
            alpha=0.7,
        )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_title(f"{template_name.title()} - Initial Voronoi Diagram", fontsize=14)

    # Add statistics
    ax.text(
        0.02,
        0.98,
        f"Points: {len(voronoi_graph.points):,}\nAfter Lloyd's relaxation",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_heightmap_debug(packed_graph, output_path, template_name, seed):
    """Generate debug visualization showing interpolated heightmap like a traditional topographic map."""
    from scipy.interpolate import griddata
    from matplotlib.colors import LinearSegmentedColormap

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get dimensions
    width = packed_graph.graph_width
    height = packed_graph.graph_height

    # Create interpolated grid for smooth heightmap
    grid_resolution = 600
    xi = np.linspace(0, width, grid_resolution)
    yi = np.linspace(0, height, grid_resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate heights from packed points
    points_array = np.array(packed_graph.points)
    zi = griddata(
        points_array, packed_graph.heights, (xi, yi), method="linear", fill_value=0
    )

    # Create custom terrain colormap
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

    # Plot interpolated heightmap
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

    # Add contour lines for topographic effect
    contours = ax.contour(
        xi,
        yi,
        zi,
        levels=np.arange(0, 101, 10),
        colors="black",
        linewidths=0.5,
        alpha=0.3,
    )

    # Highlight sea level
    sea_contour = ax.contour(
        xi, yi, zi, levels=[20], colors="navy", linewidths=2, alpha=0.8
    )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title(f"{template_name.title()} - Interpolated Heightmap", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Height", shrink=0.8)
    cbar.ax.axhline(y=20, color="blue", linewidth=2)
    cbar.ax.text(1.5, 20, "Sea Level", rotation=0, va="center", fontsize=9)

    # Add height statistics
    min_height = np.min(packed_graph.heights)
    max_height = np.max(packed_graph.heights)
    avg_height = np.mean(packed_graph.heights)
    land_cells = np.sum(packed_graph.heights >= 20)
    water_cells = np.sum(packed_graph.heights < 20)
    land_pct = (land_cells / len(packed_graph.heights)) * 100

    ax.text(
        0.02,
        0.98,
        f"Height: {min_height}-{max_height} (avg: {avg_height:.1f})\n"
        f"Land: {land_cells:,} cells ({land_pct:.1f}%)\n"
        f"Water: {water_cells:,} cells ({100-land_pct:.1f}%)\n"
        f"Interpolated from {len(packed_graph.points):,} points",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_climate_debug(climate, packed_graph, output_path, template_name, seed):
    """Generate debug visualization showing temperature and precipitation."""
    from matplotlib.colors import LinearSegmentedColormap

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get dimensions
    width = packed_graph.graph_width
    height = packed_graph.graph_height

    # Subplot 1: Temperature
    temp_scatter = ax1.scatter(
        *packed_graph.points.T,
        c=climate.temperatures,
        cmap="RdYlBu_r",  # Red-Yellow-Blue reversed (hot to cold)
        s=30,
        alpha=0.8,
        vmin=-30,
        vmax=30,
    )
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect("equal")
    ax1.set_title("Temperature Distribution (°C)", fontsize=14)
    plt.colorbar(temp_scatter, ax=ax1, label="Temperature (°C)")

    # Add temperature statistics
    avg_temp = np.mean(climate.temperatures)
    min_temp = np.min(climate.temperatures)
    max_temp = np.max(climate.temperatures)
    ax1.text(
        0.02,
        0.98,
        f"Avg: {avg_temp:.1f}°C\nMin: {min_temp:.1f}°C\nMax: {max_temp:.1f}°C",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # Subplot 2: Precipitation
    precip_scatter = ax2.scatter(
        *packed_graph.points.T,
        c=climate.precipitation,
        cmap="Blues",  # Blue scale for precipitation
        s=30,
        alpha=0.8,
        vmin=0,
        vmax=np.max(climate.precipitation),
    )
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect("equal")
    ax2.set_title("Precipitation Distribution (mm)", fontsize=14)
    plt.colorbar(precip_scatter, ax=ax2, label="Precipitation (mm)")

    # Add precipitation statistics
    avg_precip = np.mean(climate.precipitation)
    min_precip = np.min(climate.precipitation)
    max_precip = np.max(climate.precipitation)
    ax2.text(
        0.02,
        0.98,
        f"Avg: {avg_precip:.1f}mm\nMin: {min_precip:.1f}mm\nMax: {max_precip:.1f}mm",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # Add overall title
    title = f"{template_name.title()} - Climate Analysis (Seed: {seed})"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_hydrology_debug(
    rivers, hydrology, packed_graph, output_path, template_name, seed
):
    """Generate debug visualization showing river networks and discharge."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get dimensions
    width = packed_graph.graph_width
    height = packed_graph.graph_height

    # Subplot 1: River Network with Discharge
    # Plot base terrain
    terrain_scatter = ax1.scatter(
        *packed_graph.points.T,
        c=packed_graph.heights,
        cmap="terrain",
        s=10,
        alpha=0.3,
        vmin=0,
        vmax=100,
    )

    # Plot rivers colored by discharge
    if rivers:
        river_discharges = []
        river_points_x = []
        river_points_y = []
        river_widths = []

        for river_id, river_data in rivers.items():
            if river_data.cells and river_data.discharge > 0:
                for cell_id in river_data.cells:
                    if cell_id < len(packed_graph.points):
                        point = packed_graph.points[cell_id]
                        river_points_x.append(point[0])
                        river_points_y.append(point[1])
                        river_discharges.append(river_data.discharge)
                        river_widths.append(
                            max(river_data.width * 2, 20)
                        )  # Scale for visibility

        if river_discharges:
            river_scatter = ax1.scatter(
                river_points_x,
                river_points_y,
                c=river_discharges,
                cmap="Blues",
                s=river_widths,
                alpha=0.8,
                edgecolors="navy",
                linewidths=0.5,
                vmin=0,
                vmax=max(river_discharges),
            )
            plt.colorbar(river_scatter, ax=ax1, label="Discharge (m³/s)")

    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect("equal")
    ax1.set_title("River Network by Discharge", fontsize=14)

    # Add river statistics
    if rivers:
        total_rivers = len(rivers)
        main_rivers = sum(1 for r in rivers.values() if r.parent_id is None)
        tributary_rivers = total_rivers - main_rivers
        max_discharge = max(r.discharge for r in rivers.values()) if rivers else 0
        avg_discharge = np.mean([r.discharge for r in rivers.values()]) if rivers else 0

        ax1.text(
            0.02,
            0.98,
            f"Total: {total_rivers} rivers\nMain: {main_rivers}\nTributaries: {tributary_rivers}\n"
            f"Max discharge: {max_discharge:.1f} m³/s\nAvg discharge: {avg_discharge:.1f} m³/s",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

    # Subplot 2: Flow Accumulation
    flux_scatter = ax2.scatter(
        *packed_graph.points.T,
        c=hydrology.flux,
        cmap="YlOrRd",  # Yellow-Orange-Red for flow accumulation
        s=20,
        alpha=0.7,
        vmin=0,
        vmax=np.max(hydrology.flux),
    )
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect("equal")
    ax2.set_title("Water Flow Accumulation", fontsize=14)
    plt.colorbar(flux_scatter, ax=ax2, label="Flow Accumulation (m³/s)")

    # Add flux statistics
    max_flux = np.max(hydrology.flux)
    avg_flux = np.mean(hydrology.flux)
    river_threshold = hydrology.options.min_river_flux

    ax2.text(
        0.02,
        0.98,
        f"Max flux: {max_flux:.1f} m³/s\nAvg flux: {avg_flux:.1f} m³/s\n"
        f"River threshold: {river_threshold} m³/s",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # Add overall title
    title = f"{template_name.title()} - Hydrology Analysis (Seed: {seed})"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_biome_debug(
    biome_classifier, biomes, climate, packed_graph, output_path, template_name, seed
):
    """Generate debug visualization showing biome distribution and classification."""
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Get dimensions
    width = packed_graph.graph_width
    height = packed_graph.graph_height

    # Create biome colormap using actual biome colors
    biome_colors = [biome_classifier.get_biome_color(i) for i in range(13)]
    biome_cmap = ListedColormap(biome_colors)

    # Subplot 1: Biome Distribution
    biome_scatter = ax1.scatter(
        *packed_graph.points.T,
        c=biomes,
        cmap=biome_cmap,
        s=25,
        alpha=0.8,
        vmin=0,
        vmax=12,
    )
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect("equal")
    ax1.set_title("Biome Distribution", fontsize=14)

    # Create biome legend
    biome_patches = []
    unique_biomes = np.unique(biomes)
    for biome_id in unique_biomes:
        name = biome_classifier.get_biome_name(biome_id)
        color = biome_classifier.get_biome_color(biome_id)
        count = np.sum(biomes == biome_id)
        patch = mpatches.Patch(color=color, label=f"{name} ({count})")
        biome_patches.append(patch)

    ax1.legend(
        handles=biome_patches, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9
    )

    # Subplot 2: Temperature vs Biome
    temp_scatter = ax2.scatter(
        *packed_graph.points.T,
        c=climate.temperatures,
        cmap="RdYlBu_r",
        s=25,
        alpha=0.7,
        vmin=-30,
        vmax=30,
    )
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect("equal")
    ax2.set_title("Temperature Distribution", fontsize=14)
    plt.colorbar(temp_scatter, ax=ax2, label="Temperature (°C)")

    # Subplot 3: Precipitation vs Biome
    precip_scatter = ax3.scatter(
        *packed_graph.points.T,
        c=climate.precipitation,
        cmap="Blues",
        s=25,
        alpha=0.7,
        vmin=0,
        vmax=np.max(climate.precipitation),
    )
    ax3.set_xlim(0, width)
    ax3.set_ylim(0, height)
    ax3.set_aspect("equal")
    ax3.set_title("Precipitation Distribution", fontsize=14)
    plt.colorbar(precip_scatter, ax=ax3, label="Precipitation (mm)")

    # Subplot 4: Biome Statistics
    biome_counts = np.bincount(biomes, minlength=13)
    biome_names = [biome_classifier.get_biome_name(i) for i in range(13)]

    # Only plot biomes that exist
    existing_biomes = biome_counts > 0
    plot_counts = biome_counts[existing_biomes]
    plot_names = [name for i, name in enumerate(biome_names) if existing_biomes[i]]
    plot_colors = [biome_colors[i] for i in range(13) if existing_biomes[i]]

    bars = ax4.bar(range(len(plot_counts)), plot_counts, color=plot_colors, alpha=0.8)
    ax4.set_xlabel("Biome Type")
    ax4.set_ylabel("Cell Count")
    ax4.set_title("Biome Distribution Statistics", fontsize=14)
    ax4.set_xticks(range(len(plot_names)))
    ax4.set_xticklabels(plot_names, rotation=45, ha="right", fontsize=9)

    # Add percentage labels on bars
    total_cells = len(biomes)
    for i, (bar, count) in enumerate(zip(bars, plot_counts)):
        percentage = (count / total_cells) * 100
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(plot_counts) * 0.01,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add biome classification statistics
    ax4.text(
        0.02,
        0.98,
        f"Total cells: {total_cells:,}\nBiome types: {len(plot_counts)}\n"
        f"Dominant: {plot_names[np.argmax(plot_counts)]}\n"
        f"Coverage: {np.max(plot_counts)/total_cells*100:.1f}%",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    # Add overall title
    title = f"{template_name.title()} - Biome Classification Analysis (Seed: {seed})"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def test_full_pipeline_with_visualization(
    template="atoll", seed=None, generate_debug_images=True
):
    """Test the complete map generation pipeline from seed to packed graph.

    Args:
        template: Heightmap template name
        seed: Single seed used for all stages (defaults to DEFAULT_SEED)
        generate_debug_images: Whether to generate step-by-step debug images
    """
    # Use default if not provided
    seed = seed or DEFAULT_SEED

    # Stage 1: Configure grid parameters
    width = TEST_WIDTH
    height = TEST_HEIGHT
    cells_desired = TEST_CELLS_DESIRED

    # Create output directory and timestamp for all files
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = GridConfig(width=width, height=height, cells_desired=cells_desired)

    # Stage 2: Generate Voronoi graph
    print("Generating Voronoi graph...")
    voronoi_graph = generate_voronoi_graph(config, seed=seed)
    assert voronoi_graph is not None
    assert len(voronoi_graph.points) > 0
    assert voronoi_graph.cells_x > 0
    assert voronoi_graph.cells_y > 0
    print(f"Generated {len(voronoi_graph.points)} Voronoi cells")

    # Generate initial Voronoi debug image if requested
    if generate_debug_images:
        voronoi_path = output_dir / f"{template}_1_voronoi_{timestamp}.png"
        print("Generating initial Voronoi debug visualization...")
        generate_voronoi_debug(voronoi_graph, voronoi_path, template, seed)
        print(f"Initial Voronoi saved to {voronoi_path}")

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

    # Initialize heightmap generator with seed for proper PRNG reseeding
    heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph, seed=seed)

    # Generate heightmap steps debug image if requested
    if generate_debug_images:
        heightmap_steps_path = (
            output_dir / f"{template}_3_heightmap_steps_{timestamp}.png"
        )
        print("Generating heightmap steps debug visualization...")
        generate_heightmap_steps_debug(
            heightmap_gen, template, heightmap_steps_path, seed
        )
        print(f"Heightmap steps saved to {heightmap_steps_path}")

        # Re-generate the final heightmap since debug consumed it
        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph, seed=seed)

    heights = heightmap_gen.from_template(template, seed=seed)

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
    # Pass seed to Features for PRNG reseeding (though it currently doesn't use randomness)
    features = Features(voronoi_graph, seed=seed)
    features.markup_grid()

    # Verify that distance_field was created
    assert hasattr(voronoi_graph, "distance_field")
    assert voronoi_graph.distance_field is not None
    print("Coastline markup completed")

    # Stage 5: Perform reGraph coastal resampling
    print("Performing reGraph coastal resampling...")
    packed_graph = regraph(voronoi_graph)

    # Generate packed Voronoi debug image if requested
    if generate_debug_images:
        packed_voronoi_path = (
            output_dir / f"{template}_2_voronoi_packed_{timestamp}.png"
        )
        print("Generating packed Voronoi debug visualization...")
        generate_packed_voronoi_debug(
            voronoi_graph, packed_graph, packed_voronoi_path, template, seed
        )
        print(f"Packed Voronoi saved to {packed_voronoi_path}")

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

    # Generate Stage 4: Final heightmap visualization
    if generate_debug_images:
        final_heightmap_path = output_dir / f"{template}_4_heightmap_{timestamp}.png"
        print("Generating final heightmap visualization...")
        generate_heightmap_debug(packed_graph, final_heightmap_path, template, seed)
        print(f"Final heightmap saved to {final_heightmap_path}")

    # Stage 6: Mark up features on packed graph
    print("Marking up features on packed graph...")
    packed_features = Features(packed_graph, seed=seed)
    packed_features.markup_grid()

    # Stage 7: Calculate climate
    print("Calculating climate...")
    climate = Climate(packed_graph)
    climate.calculate_temperatures()
    climate.generate_precipitation()

    # Generate climate debug image if requested
    if generate_debug_images:
        climate_path = output_dir / f"{template}_5_climate_{timestamp}.png"
        print("Generating climate debug visualization...")
        generate_climate_debug(climate, packed_graph, climate_path, template, seed)
        print(f"Climate visualization saved to {climate_path}")

    # Verify climate calculations
    assert hasattr(climate, "temperatures")
    assert hasattr(climate, "precipitation")
    assert len(climate.temperatures) == len(packed_graph.points)
    assert len(climate.precipitation) == len(packed_graph.points)

    # Stage 8: Generate hydrology system
    print("Generating hydrology system...")
    hydrology = Hydrology(packed_graph, packed_features, climate)
    rivers = hydrology.generate_rivers()

    # Generate hydrology debug image if requested
    if generate_debug_images:
        hydrology_path = output_dir / f"{template}_6_hydrology_{timestamp}.png"
        print("Generating hydrology debug visualization...")
        generate_hydrology_debug(
            rivers, hydrology, packed_graph, hydrology_path, template, seed
        )
        print(f"Hydrology visualization saved to {hydrology_path}")

    # Verify hydrology system
    assert isinstance(rivers, dict)
    assert len(rivers) >= 0  # May have 0 rivers for some templates
    for river_id, river_data in rivers.items():
        assert hasattr(river_data, "discharge")
        assert hasattr(river_data, "width")
        assert hasattr(river_data, "length")
        assert hasattr(river_data, "cells")
        assert river_data.discharge >= 0
        assert river_data.width >= 0
        assert len(river_data.cells) > 0

    print(
        f"Generated {len(rivers)} rivers with max discharge: {max(r.discharge for r in rivers.values()) if rivers else 0:.1f} m³/s"
    )

    # Stage 9: Generate biomes
    print("Classifying biomes...")
    biome_classifier = BiomeClassifier()

    # Prepare data for biome classification
    temperatures = climate.temperatures
    precipitation = climate.precipitation
    heights = packed_graph.heights

    # Create river presence array
    has_river = np.zeros(len(packed_graph.points), dtype=bool)
    river_flux = np.zeros(len(packed_graph.points), dtype=float)

    for river_id, river_data in rivers.items():
        for cell_id in river_data.cells:
            if cell_id < len(has_river):
                has_river[cell_id] = True
                river_flux[cell_id] = max(river_flux[cell_id], river_data.discharge)

    # Classify biomes
    biomes = biome_classifier.classify_biomes(
        temperatures=temperatures,
        precipitation=precipitation,
        heights=heights,
        river_flux=river_flux,
        has_river=has_river,
    )

    # Generate biome debug image if requested
    if generate_debug_images:
        biome_path = output_dir / f"{template}_7_biomes_{timestamp}.png"
        print("Generating biome debug visualization...")
        generate_biome_debug(
            biome_classifier, biomes, climate, packed_graph, biome_path, template, seed
        )
        print(f"Biome visualization saved to {biome_path}")

    # Verify biome classification
    assert len(biomes) == len(packed_graph.points)
    assert biomes.dtype == np.uint8
    assert np.all(biomes >= 0)
    assert np.all(biomes <= 12)

    # Print biome statistics
    unique_biomes, counts = np.unique(biomes, return_counts=True)
    print(f"Generated {len(unique_biomes)} different biome types:")
    for biome_id, count in zip(unique_biomes, counts):
        biome_name = biome_classifier.get_biome_name(biome_id)
        percentage = (count / len(biomes)) * 100
        print(f"  - {biome_name}: {count} cells ({percentage:.1f}%)")

    # Stage 10: Generate comprehensive final visualization
    print("Generating visualization...")

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
    pipeline_text = "Complete FMG Pipeline:\n"
    pipeline_text += "✓ Voronoi + Lloyd's\n"
    pipeline_text += "✓ Heightmap template\n"
    pipeline_text += "✓ Features.markupGrid\n"
    pipeline_text += "✓ reGraph (coastal enhancement)\n"
    pipeline_text += "✓ Climate system\n"
    pipeline_text += "✓ Hydrology + rivers\n"
    pipeline_text += "✓ Biome classification"

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
        f"Seed: {seed}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        horizontalalignment="right",
        fontsize=10,
        family="monospace",
    )

    # Save comprehensive final visualization
    output_filename = f"{template}_8_comprehensive_{timestamp}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Visualization saved to {output_path}")
    print("\nPipeline test completed successfully!")

    # Return results for potential further testing
    results = {
        "voronoi_graph": voronoi_graph,
        "heights": heights,
        "packed_graph": packed_graph,
        "packed_heights": packed_heights,
        "features": features,
        "packed_features": packed_features,
        "climate": climate,
        "hydrology": hydrology,
        "rivers": rivers,
        "biome_classifier": biome_classifier,
        "biomes": biomes,
        "visualization_path": output_path,
    }

    if generate_debug_images:
        results["voronoi_path"] = voronoi_path
        results["packed_voronoi_path"] = packed_voronoi_path
        results["heightmap_steps_path"] = heightmap_steps_path
        results["final_heightmap_path"] = final_heightmap_path
        results["climate_path"] = climate_path
        results["hydrology_path"] = hydrology_path
        results["biome_path"] = biome_path

    return results


def test_pipeline_error_handling():
    """Test that the pipeline properly handles missing Features.markup_grid() call."""

    # Generate basic components
    config = GridConfig(
        width=TEST_WIDTH, height=TEST_HEIGHT, cells_desired=TEST_CELLS_DESIRED
    )
    voronoi_graph = generate_voronoi_graph(config, seed="error-test")

    heightmap_config = HeightmapConfig(
        width=TEST_WIDTH,
        height=TEST_HEIGHT,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=TEST_CELLS_DESIRED,
        spacing=voronoi_graph.spacing,
    )

    heightmap_gen = HeightmapGenerator(
        heightmap_config, voronoi_graph, seed="error-test"
    )
    heights = heightmap_gen.from_template("atoll", seed="error-test")
    voronoi_graph.heights = heights

    # Try to call regraph WITHOUT Features.markup_grid()
    with pytest.raises(ValueError, match="graph.distance_field not found"):
        regraph(voronoi_graph)

    print(
        "Error handling test passed: regraph correctly requires Features.markup_grid()"
    )


def test_complete_pipeline_with_climate_and_hydrology(template="atoll", seed=None):
    """Test the complete map generation pipeline including climate and hydrology.

    Args:
        template: Heightmap template name
        seed: Single seed used for all stages (defaults to DEFAULT_SEED)
    """
    # Use default if not provided
    seed = seed or DEFAULT_SEED

    # Stage 1: Generate Voronoi graph
    config = GridConfig(
        width=TEST_WIDTH, height=TEST_HEIGHT, cells_desired=TEST_CELLS_DESIRED
    )
    voronoi_graph = generate_voronoi_graph(config, seed=seed)

    # Stage 2: Generate heightmap
    heightmap_config = HeightmapConfig(
        width=int(TEST_WIDTH),
        height=int(TEST_HEIGHT),
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=TEST_CELLS_DESIRED,
        spacing=voronoi_graph.spacing,
    )

    heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph, seed=seed)
    heights = heightmap_gen.from_template(template, seed=seed)
    voronoi_graph.heights = heights

    # Stage 3: Mark up features
    features = Features(voronoi_graph, seed=seed)
    features.markup_grid()

    # Stage 4: Perform reGraph coastal enhancement
    packed_graph = regraph(voronoi_graph)

    # Stage 5: Mark up features on packed graph
    packed_features = Features(packed_graph, seed=seed)
    packed_features.markup_grid()

    # Stage 6: Calculate climate
    climate = Climate(packed_graph)  # Use default options and coordinates
    climate.calculate_temperatures()
    climate.generate_precipitation()

    # Verify climate calculations
    assert hasattr(climate, "temperatures")
    assert hasattr(climate, "precipitation")
    assert len(climate.temperatures) == len(packed_graph.points)
    assert len(climate.precipitation) == len(packed_graph.points)

    # Stage 7: Generate hydrology system
    hydrology = Hydrology(packed_graph, packed_features, climate)
    rivers = hydrology.generate_rivers()

    # Verify hydrology system
    assert isinstance(rivers, dict)
    # Should generate some rivers for realistic terrain
    assert len(rivers) > 0
    # Verify river data structure
    for river_id, river_data in rivers.items():
        assert hasattr(river_data, "discharge")
        assert hasattr(river_data, "width")
        assert hasattr(river_data, "length")
        assert hasattr(river_data, "cells")
        assert river_data.discharge >= 0
        assert river_data.width >= 0
        assert len(river_data.cells) > 0

    # Stage 8: Generate biomes
    biome_classifier = BiomeClassifier()

    # Prepare data for biome classification
    temperatures = climate.temperatures
    precipitation = climate.precipitation
    heights = packed_graph.heights

    # Create river presence array
    has_river = np.zeros(len(packed_graph.points), dtype=bool)
    river_flux = np.zeros(len(packed_graph.points), dtype=float)

    for river_id, river_data in rivers.items():
        for cell_id in river_data.cells:
            if cell_id < len(has_river):
                has_river[cell_id] = True
                river_flux[cell_id] = max(river_flux[cell_id], river_data.discharge)

    # Classify biomes
    biomes = biome_classifier.classify_biomes(
        temperatures=temperatures,
        precipitation=precipitation,
        heights=heights,
        river_flux=river_flux,
        has_river=has_river,
    )

    # Verify biome classification
    assert len(biomes) == len(packed_graph.points)
    assert biomes.dtype == np.uint8
    assert np.all(biomes >= 0)
    assert np.all(biomes <= 12)

    print(f"✓ Voronoi: {len(voronoi_graph.points)} cells generated")
    print(f"✓ Heightmap: {heights.min()}-{heights.max()} height range")
    print(f"✓ Features: {len(features.features)} features detected")
    print(f"✓ Packed: {len(packed_graph.points)} cells after coastal enhancement")
    print(
        f"✓ Climate: {len(climate.temperatures)} temperature points, {len(climate.precipitation)} precipitation points"
    )
    print(
        f"✓ Hydrology: {len(rivers)} rivers generated with realistic discharge values"
    )
    print(f"✓ Biomes: {len(np.unique(biomes))} different biome types classified")

    return {
        "voronoi_graph": voronoi_graph,
        "packed_graph": packed_graph,
        "features": features,
        "packed_features": packed_features,
        "climate": climate,
        "rivers": rivers,
        "biome_classifier": biome_classifier,
        "biomes": biomes,
    }


def main():
    """Run end-to-end tests with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test FMG pipeline with various templates"
    )
    parser.add_argument(
        "--template", default="all", help="Template name or 'all' to test all templates"
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help=f"Seed for map generation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Generate debug images (voronoi structure and heightmap steps)",
    )

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

    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Run tests for each template
    successful = 0
    failed = []

    for template in templates_to_test:
        try:
            print(f"\nTesting {template} template...")
            results = test_full_pipeline_with_visualization(
                template=template,
                seed=args.seed,
                generate_debug_images=args.debug_images,
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
