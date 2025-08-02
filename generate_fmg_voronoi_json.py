#!/usr/bin/env python3
"""
Generate a Voronoi graph with the exact FMG export format.
This matches the structure from FMG's exportVoronoiData() function.
"""

import json
import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def voronoi_to_fmg_export_format(graph, seed, width, height):
    """Convert VoronoiGraph to FMG's exact export format."""
    # This matches the format from FMG's exportVoronoiData() function
    json_data = {
        "seed": str(seed),
        "graphWidth": width,
        "graphHeight": height,
        "voronoi": {
            "cells": {
                "p": graph.points.tolist(),  # Cell positions
                "g": None,  # grid - only used for packed cells
                "h": graph.heights.tolist(),  # Heights
                "c": convert_to_serializable(graph.cell_neighbors),  # Cell neighbors
                "v": convert_to_serializable(graph.cell_vertices),  # Cell vertices
                "b": graph.cell_border_flags.tolist(),  # Border flags
            },
            "vertices": {
                "p": graph.vertex_coordinates.tolist(),  # Vertex positions
                "v": convert_to_serializable(
                    graph.vertex_neighbors
                ),  # Vertex neighbors
                "c": convert_to_serializable(graph.vertex_cells),  # Vertex cells
            },
        },
    }

    # Add optional fields that FMG includes when available
    if hasattr(graph, "cell_types") and graph.cell_types is not None:
        json_data["voronoi"]["cells"]["t"] = graph.cell_types.tolist()

    if hasattr(graph, "feature_ids") and graph.feature_ids is not None:
        json_data["voronoi"]["cells"]["f"] = graph.feature_ids.tolist()

    if hasattr(graph, "cell_areas") and graph.cell_areas is not None:
        json_data["voronoi"]["cells"]["area"] = graph.cell_areas.tolist()

    # Add spacing info as FMG does
    json_data["spacing"] = float(graph.spacing)
    json_data["cellsX"] = int(graph.cells_x)
    json_data["cellsY"] = int(graph.cells_y)

    return json_data


def main():
    # FMG parameters
    seed = "162921633"
    width = 1200
    height = 1000
    cells_desired = 10000  # FMG default

    # Create grid configuration
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)

    # Generate Voronoi graph
    print(
        f"Generating Voronoi graph with seed={seed}, width={width}, height={height}, cells={cells_desired}"
    )
    graph = generate_voronoi_graph(config, seed=seed, apply_relaxation=True)

    # Convert to FMG export format
    json_data = voronoi_to_fmg_export_format(graph, seed, width, height)

    # Save to file
    output_filename = f"py-fmg_voronoi_{seed}.json"
    with open(output_filename, "w") as f:
        # Use same formatting as JavaScript JSON.stringify
        json.dump(json_data, f, indent=2, separators=(",", ": "))

    print(f"Voronoi graph saved to {output_filename}")
    print(f"Generated {len(graph.points)} cells")
    print(f"Grid dimensions: {graph.cells_x} x {graph.cells_y}")
    print(f"Spacing: {graph.spacing}")


if __name__ == "__main__":
    main()
