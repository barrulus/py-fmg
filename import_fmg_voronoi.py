#!/usr/bin/env python3
"""
Import FMG Voronoi data and convert it to our format for compatibility testing.
This allows you to use FMG-generated Voronoi data with our Python implementation.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VoronoiGraph:
    """Minimal VoronoiGraph compatible with FMG imported data."""

    # Grid parameters
    spacing: float
    cells_desired: int
    graph_width: float
    graph_height: float
    seed: str

    # Points data
    boundary_points: np.ndarray
    points: np.ndarray
    cells_x: int
    cells_y: int

    # Cell connectivity data
    cell_neighbors: List[List[int]]
    cell_vertices: List[List[int]]
    cell_border_flags: np.ndarray
    heights: np.ndarray

    # Vertex data
    vertex_coordinates: np.ndarray
    vertex_neighbors: List[List[int]]
    vertex_cells: List[List[int]]

    # Optional fields
    grid_indices: Optional[np.ndarray] = field(default=None)
    distance_field: Optional[np.ndarray] = field(default=None)
    feature_ids: Optional[np.ndarray] = field(default=None)
    features: Optional[List] = field(default=None)
    border_cells: Optional[np.ndarray] = field(default=None)


def import_fmg_voronoi(json_file_path):
    """Import FMG Voronoi JSON and convert to VoronoiGraph object."""
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract basic parameters
    seed = data["seed"]
    width = data["graphWidth"]
    height = data["graphHeight"]

    voronoi_data = data["voronoi"]
    cells = voronoi_data["cells"]
    vertices = voronoi_data["vertices"]

    # Convert cell data
    points = np.array(cells["p"])
    cell_neighbors = cells["c"]
    cell_vertices = cells["v"]
    cell_border_flags = np.array(cells["b"])
    heights = np.array(cells["h"])

    # Convert vertex data
    vertex_coordinates = np.array(vertices["p"])
    vertex_neighbors = vertices["v"]
    vertex_cells = vertices["c"]

    # Calculate derived parameters
    n_cells = len(points)
    spacing = data.get("spacing", np.sqrt(width * height / n_cells))
    cells_x = data.get("cellsX", int(width / spacing))
    cells_y = data.get("cellsY", int(height / spacing))

    # Create empty boundary points array (not included in FMG export)
    boundary_points = np.array([])

    # Create VoronoiGraph object
    graph = VoronoiGraph(
        spacing=spacing,
        cells_desired=n_cells,
        graph_width=width,
        graph_height=height,
        seed=seed,
        boundary_points=boundary_points,
        points=points,
        cells_x=cells_x,
        cells_y=cells_y,
        cell_neighbors=cell_neighbors,
        cell_vertices=cell_vertices,
        cell_border_flags=cell_border_flags,
        heights=heights,
        vertex_coordinates=vertex_coordinates,
        vertex_neighbors=vertex_neighbors,
        vertex_cells=vertex_cells,
    )

    # Add optional fields if present
    if "t" in cells:
        graph.distance_field = np.array(cells["t"])
    if "f" in cells:
        graph.feature_ids = np.array(cells["f"])
    if "g" in cells:
        graph.grid_indices = np.array(cells["g"])

    return graph


def main():
    """Example usage: Import FMG data and use it."""
    # You would need to save the FMG JSON data to a file first
    fmg_json_file = "voronoi-data-162921633.json"

    print(f"Note: To use this script, you need to:")
    print(f"1. Export Voronoi data from FMG using the browser console")
    print(f"2. Save it as '{fmg_json_file}'")
    print(f"3. Run this script again")
    print()
    print("Example FMG export code:")
    print("```javascript")
    print("// In FMG browser console:")
    print("const data = {")
    print('  seed: "162921633",')
    print("  graphWidth: 1200,")
    print("  graphHeight: 1000,")
    print("  voronoi: {")
    print("    cells: pack.cells,")
    print("    vertices: pack.vertices")
    print("  }")
    print("};")
    print(
        "saveStringToFile(JSON.stringify(data, null, 2), 'fmg_export_162921633.json');"
    )
    print("```")

    # Try to import if file exists
    import os

    if os.path.exists(fmg_json_file):
        graph = import_fmg_voronoi(fmg_json_file)
        print(f"\nSuccessfully imported FMG Voronoi data:")
        print(f"  Seed: {graph.seed}")
        print(f"  Dimensions: {graph.graph_width} x {graph.graph_height}")
        print(f"  Cells: {len(graph.points)}")
        print(f"  Vertices: {len(graph.vertex_coordinates)}")
        print(f"  First 5 points:")
        for i in range(min(5, len(graph.points))):
            print(f"    {i}: {graph.points[i]}")


if __name__ == "__main__":
    main()
