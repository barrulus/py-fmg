#!/usr/bin/env python3
"""Visualize a generated map from the database."""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import requests

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value

from sqlalchemy import create_engine, text


def get_latest_map_id():
    """Get the ID of the most recently generated map via API."""
    try:
        response = requests.get("http://localhost:8000/maps")
        if response.status_code == 200:
            maps = response.json()
            if maps:
                return maps[0]["id"]  # Most recent map
    except Exception as e:
        print(f"Could not get map from API: {e}")
    return None


def visualize_map(map_id=None):
    """Visualize a map from the database."""
    if not map_id:
        map_id = get_latest_map_id()
        if not map_id:
            print("No maps found or API not available")
            return

    print(f"Visualizing map: {map_id}")

    # Connect to database
    db_url = f'postgresql://{os.environ["DB_USER"]}:{os.environ["DB_PASSWORD"]}@{os.environ["DB_HOST"]}:{os.environ["DB_PORT"]}/{os.environ["DB_NAME"]}'
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Get map metadata
        result = conn.execute(
            text(
                """
            SELECT name, seed, width, height, cells_count, created_at 
            FROM maps WHERE id = :map_id
        """
            ),
            {"map_id": map_id},
        )

        map_data = result.fetchone()
        if not map_data:
            print(f"Map {map_id} not found in database")
            return

        name, seed, width, height, cells_count, created_at = map_data
        print(f"Map: {name}")
        print(f"Seed: {seed}")
        print(f"Size: {width}x{height}")
        print(f"Cells: {cells_count}")
        print(f"Created: {created_at}")

        # Get Voronoi cells
        result = conn.execute(
            text(
                """
            SELECT cell_index, height, is_land, center_x, center_y,
                   ST_X(ST_Centroid(geometry)) as geom_x, ST_Y(ST_Centroid(geometry)) as geom_y
            FROM voronoi_cells WHERE map_id = :map_id
            ORDER BY cell_index
        """
            ),
            {"map_id": map_id},
        )

        voronoi_cells = result.fetchall()
        print(f"Voronoi cells: {len(voronoi_cells)}")

        # Get settlements
        result = conn.execute(
            text(
                """
            SELECT name, ST_X(geometry) as x, ST_Y(geometry) as y, 
                   settlement_type, population, is_capital
            FROM settlements WHERE map_id = :map_id
        """
            ),
            {"map_id": map_id},
        )

        settlements = result.fetchall()
        print(f"Settlements: {len(settlements)}")

        # Get rivers (if any)
        result = conn.execute(
            text(
                """
            SELECT name, ST_AsText(geometry) as geom_text, length_km, discharge_m3s
            FROM rivers WHERE map_id = :map_id
        """
            ),
            {"map_id": map_id},
        )

        rivers = result.fetchall()
        print(f"Rivers: {len(rivers)}")

        # Get state/culture boundaries
        result = conn.execute(
            text(
                """
            SELECT name, color, ST_AsText(geometry) as geom_text
            FROM cultures WHERE map_id = :map_id AND geometry IS NOT NULL
        """
            ),
            {"map_id": map_id},
        )

        cultures = result.fetchall()
        print(f"Cultures with territories: {len(cultures)}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    # Set map bounds
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    # Plot Voronoi cells colored by height
    if voronoi_cells:
        # Extract data for plotting
        xs = [cell[3] for cell in voronoi_cells]  # center_x
        ys = [cell[4] for cell in voronoi_cells]  # center_y
        heights = [cell[1] for cell in voronoi_cells]  # height
        is_land = [cell[2] for cell in voronoi_cells]  # is_land

        # Create scatter plot colored by height
        scatter = ax.scatter(xs, ys, c=heights, cmap="terrain", s=30, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label="Height")

        # Add land/ocean info
        land_count = sum(is_land)
        ocean_count = len(is_land) - land_count
        print(f"Land cells: {land_count}, Ocean cells: {ocean_count}")
    else:
        # Fallback: Create a background representing the map area
        ax.add_patch(
            plt.Rectangle((0, 0), width, height, facecolor="lightblue", alpha=0.3)
        )

    # Plot culture/state boundaries
    if cultures:
        from shapely import wkt
        import matplotlib.patches as patches

        for culture in cultures:
            culture_name, color, geom_text = culture
            try:
                # Parse the geometry
                geom = wkt.loads(geom_text)

                # Convert hex color to matplotlib format
                if color.startswith("#"):
                    face_color = color
                else:
                    face_color = f"#{color}" if len(color) == 6 else "#888888"

                # Plot polygon boundary
                if hasattr(geom, "exterior"):
                    # Single polygon
                    x, y = geom.exterior.xy
                    ax.plot(
                        x,
                        y,
                        color=face_color,
                        linewidth=2,
                        alpha=0.8,
                        label=f"{culture_name} border",
                    )
                    # Optional: fill with transparent color
                    ax.fill(x, y, color=face_color, alpha=0.1)
                elif hasattr(geom, "geoms"):
                    # MultiPolygon
                    for poly in geom.geoms:
                        if hasattr(poly, "exterior"):
                            x, y = poly.exterior.xy
                            ax.plot(x, y, color=face_color, linewidth=2, alpha=0.8)
                            ax.fill(x, y, color=face_color, alpha=0.1)
            except Exception as e:
                print(f"Could not plot culture {culture_name}: {e}")

    # Plot rivers
    if rivers:
        from shapely import wkt

        for river in rivers:
            river_name, geom_text, length_km, discharge = river
            try:
                # Parse the linestring geometry
                geom = wkt.loads(geom_text)

                # Determine river width based on discharge
                if discharge > 1000:
                    linewidth = 3.0  # Major river
                    color = "#0066CC"
                elif discharge > 100:
                    linewidth = 2.0  # Medium river
                    color = "#0080FF"
                else:
                    linewidth = 1.0  # Small river
                    color = "#00AAFF"

                # Plot the river
                if hasattr(geom, "xy"):
                    # Single linestring
                    x, y = geom.xy
                    ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.8, zorder=5)
                elif hasattr(geom, "geoms"):
                    # MultiLineString
                    for line in geom.geoms:
                        x, y = line.xy
                        ax.plot(
                            x, y, color=color, linewidth=linewidth, alpha=0.8, zorder=5
                        )
            except Exception as e:
                print(f"Could not plot river {river_name}: {e}")

    # Plot settlements
    if settlements:
        for settlement in settlements:
            settlement_name, x, y, settlement_type, population, is_capital = settlement

            # Different colors/sizes for different settlement types
            if is_capital:
                color = "red"
                size = 100
                marker = "s"  # square
            elif settlement_type and "city" in settlement_type.lower():
                color = "orange"
                size = 60
                marker = "o"
            else:
                color = "brown"
                size = 30
                marker = "o"

            ax.scatter(
                x,
                y,
                c=color,
                s=size,
                marker=marker,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add settlement name (for larger settlements)
            if size >= 60:
                ax.annotate(
                    settlement_name,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

    # Add title and info
    ax.set_title(
        f"{name}\nSeed: {seed} | Cells: {cells_count} | Settlements: {len(settlements)}",
        fontsize=14,
        pad=20,
    )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        plt.scatter([], [], c="red", s=100, marker="s", label="Capital"),
        plt.scatter([], [], c="orange", s=60, marker="o", label="City"),
        plt.scatter([], [], c="brown", s=30, marker="o", label="Town/Village"),
        Line2D([0], [0], color="#0066CC", linewidth=3, label="Major River"),
        Line2D([0], [0], color="#0080FF", linewidth=2, label="Medium River"),
        Line2D([0], [0], color="#00AAFF", linewidth=1, label="Small River"),
        Line2D([0], [0], color="gray", linewidth=2, alpha=0.8, label="Culture Border"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"map_{seed}_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Map visualization saved as: {filename}")

    plt.show()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a generated map")
    parser.add_argument(
        "--map-id", help="Map ID to visualize (uses latest if not specified)"
    )

    args = parser.parse_args()

    visualize_map(args.map_id)


if __name__ == "__main__":
    main()
