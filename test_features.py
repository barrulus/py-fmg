#!/usr/bin/env python3
"""Test the geographic features implementation."""

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.utils.random import set_random_seed


def test_features():
    """Test feature detection on a small map."""
    print("Testing geographic features implementation...")
    
    # Set seed for reproducibility
    seed = "test_features_123"
    set_random_seed(seed)
    
    # Generate small test grid
    config = GridConfig(width=200, height=200, cells_desired=500)
    graph = generate_voronoi_graph(config, seed=seed)
    
    print(f"Generated Voronoi graph with {len(graph.points)} cells")
    
    # Generate heightmap
    heightmap_config = HeightmapConfig(
        width=config.width,
        height=config.height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=config.cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    heights = generator.from_template("archipelago", seed=seed)
    
    # Print height statistics before features
    print(f"\nHeightmap statistics:")
    print(f"  Min height: {np.min(heights)}")
    print(f"  Max height: {np.max(heights)}")
    print(f"  Mean height: {np.mean(heights):.1f}")
    print(f"  Median height: {np.median(heights):.1f}")
    print(f"  Land cells (h>=20): {np.sum(heights >= 20)} ({np.sum(heights >= 20)/len(heights)*100:.1f}%)")
    print(f"  Water cells (h<20): {np.sum(heights < 20)} ({np.sum(heights < 20)/len(heights)*100:.1f}%)")
    
    # Print height distribution
    print("\nHeight distribution:")
    for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        count = np.sum(heights >= threshold)
        print(f"  h >= {threshold:3d}: {count:4d} cells ({count/len(heights)*100:5.1f}%)")
    
    # Create and run features detection
    features = Features(graph)
    features.markup_grid()
    
    print(f"\nFeatures detected: {len(features.features) - 1}")  # -1 because index 0 is reserved
    
    # Count feature types
    oceans = sum(1 for f in features.features[1:] if f.type == "ocean")
    lakes = sum(1 for f in features.features[1:] if f.type == "lake")
    islands = sum(1 for f in features.features[1:] if f.type == "island")
    
    print(f"  Oceans: {oceans}")
    print(f"  Lakes: {lakes}")
    print(f"  Islands: {islands}")
    
    # Check distance field
    coast_cells = np.sum(features.distance_field == 1)  # LAND_COAST
    water_coast = np.sum(features.distance_field == -1)  # WATER_COAST
    deep_water = np.sum(features.distance_field < -1)  # DEEP_WATER
    unmarked = np.sum(features.distance_field == 0)  # UNMARKED
    
    print(f"\nDistance field statistics:")
    print(f"  Unmarked cells: {unmarked}")
    print(f"  Land coast cells: {coast_cells}")
    print(f"  Water coast cells: {water_coast}")
    print(f"  Deep water cells: {deep_water}")
    
    # Debug: check unique values in distance field
    unique_distances = np.unique(features.distance_field)
    print(f"  Unique distance values: {unique_distances}")
    
    # Test lake detection in deep depressions
    print("\nTesting lake detection in deep depressions...")
    initial_lakes = lakes
    features.add_lakes_in_deep_depressions(elevation_limit=20)
    
    # Recount features
    new_lakes = sum(1 for f in features.features[1:] if f and f.type == "lake")
    print(f"  Lakes added: {new_lakes - initial_lakes}")
    
    # Test opening near-sea lakes
    print("\nTesting near-sea lake opening...")
    features.open_near_sea_lakes()
    
    # Final count
    final_lakes = sum(1 for f in features.features[1:] if f and f.type == "lake")
    final_oceans = sum(1 for f in features.features[1:] if f and f.type == "ocean")
    
    print(f"  Final lakes: {final_lakes}")
    print(f"  Final oceans: {final_oceans}")
    
    # Verify all cells are assigned to features
    unassigned = np.sum(features.feature_ids == 0)
    print(f"\nUnassigned cells: {unassigned}")
    
    # Print largest features
    print("\nLargest features by cell count:")
    sorted_features = sorted(
        [f for f in features.features[1:] if f], 
        key=lambda x: x.cells, 
        reverse=True
    )[:5]
    
    for f in sorted_features:
        print(f"  {f.type.title()} #{f.id}: {f.cells} cells")


if __name__ == "__main__":
    test_features()