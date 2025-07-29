#!/usr/bin/env python3
"""
Test script to verify heightmap generation issues:
1. getLinePower bug behavior
2. Blob spreading cell coverage
3. D3.scan vs np.argmin behavior
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.utils.random import set_random_seed


def test_line_power_bug():
    """Test that getLinePower always returns 0.81 (FMG bug compatibility)"""
    print("Testing getLinePower bug...")

    config = HeightmapConfig(
        width=1000, height=1000, cells_x=100, cells_y=100, cells_desired=10000
    )

    # Create minimal graph structure
    graph = type(
        "Graph",
        (),
        {"points": np.zeros((10000, 2)), "cell_neighbors": [[] for _ in range(10000)]},
    )()

    generator = HeightmapGenerator(config, graph)

    # Test various cell counts - should all return 0.81
    test_values = [1000, 5000, 10000, 50000, 100000]
    for cells in test_values:
        power = generator._get_line_power(cells)
        print(f"  cells={cells}: line_power={power}")
        assert power == 0.81, f"Expected 0.81, got {power}"

    print("✓ getLinePower bug correctly replicated\n")


def test_blob_spreading_coverage():
    """Test blob spreading to analyze cell coverage"""
    print("Testing blob spreading coverage...")

    # Use a fixed seed for reproducibility
    set_random_seed("test-blob")

    # Generate a small test grid
    config = GridConfig(width=100, height=100, cells_desired=100)
    graph = generate_voronoi_graph(config, seed="test-blob")

    config = HeightmapConfig(
        width=100, height=100, cells_x=10, cells_y=10, cells_desired=100
    )

    generator = HeightmapGenerator(config, graph)

    # Test single hill spreading
    initial_heights = generator.heights.copy()
    generator.add_hill(1, 50, "40-60", "40-60")

    # Count affected cells
    affected = np.sum(generator.heights != initial_heights)
    total = len(generator.heights)
    percentage = (affected / total) * 100

    print(f"  Affected cells: {affected}/{total} ({percentage:.1f}%)")
    print(f"  Blob power: {generator.blob_power}")

    # Analyze spread pattern
    height_changes = generator.heights - initial_heights
    non_zero = height_changes[height_changes > 0]
    if len(non_zero) > 0:
        print(f"  Min change: {np.min(non_zero):.6f}")
        print(f"  Max change: {np.max(non_zero):.1f}")
        print(f"  Mean change: {np.mean(non_zero):.2f}")

    print(f"✓ Blob spreading analysis complete\n")

    return percentage


def test_argmin_behavior():
    """Test that np.argmin behaves like D3.scan for ties"""
    print("Testing np.argmin vs D3.scan behavior...")

    # Test with ties - both should return first occurrence
    test_arrays = [
        [5, 3, 3, 7, 3],  # Multiple minimums
        [1, 1, 1, 1, 1],  # All same
        [9, 8, 7, 6, 5],  # Descending
        [1, 2, 3, 4, 5],  # Ascending
    ]

    for arr in test_arrays:
        min_idx = np.argmin(arr)
        min_val = arr[min_idx]
        first_min_idx = arr.index(min_val)
        print(f"  Array {arr}: argmin={min_idx}, first_occurrence={first_min_idx}")
        assert min_idx == first_min_idx, "np.argmin should return first occurrence"

    print("✓ np.argmin correctly returns first minimum (matches D3.scan)\n")


def test_heightmap_generation():
    """Test full heightmap generation with a template"""
    print("Testing full heightmap generation...")

    set_random_seed("test-full")

    # Generate Voronoi graph
    config = GridConfig(width=1000, height=1000, cells_desired=5000)
    graph = generate_voronoi_graph(config, seed="test-full")

    config = HeightmapConfig(
        width=1000, height=1000, cells_x=71, cells_y=71, cells_desired=5000
    )

    generator = HeightmapGenerator(config, graph)

    # Test with archipelago template
    heights = generator.from_template("archipelago")

    print(f"  Generated heights shape: {heights.shape}")
    print(f"  Height range: {np.min(heights)} - {np.max(heights)}")
    print(f"  Mean height: {np.mean(heights):.1f}")
    print(
        f"  Land cells (h>=20): {np.sum(heights >= 20)} ({np.sum(heights >= 20)/len(heights)*100:.1f}%)"
    )
    print(
        f"  Water cells (h<20): {np.sum(heights < 20)} ({np.sum(heights < 20)/len(heights)*100:.1f}%)"
    )

    # Verify heights are valid
    assert np.all(heights >= 0) and np.all(heights <= 100), "Heights out of range"
    assert heights.dtype == np.uint8, "Heights should be uint8"

    print("✓ Full heightmap generation successful\n")


def main():
    print("=" * 60)
    print("Heightmap Generation Issue Tests")
    print("=" * 60)
    print()

    # Run all tests
    test_line_power_bug()
    test_argmin_behavior()
    blob_coverage = test_blob_spreading_coverage()
    test_heightmap_generation()

    print("=" * 60)
    print("Summary:")
    print("- getLinePower bug: ✓ Correctly replicated")
    print("- np.argmin behavior: ✓ Matches D3.scan")
    print(f"- Blob spreading: {blob_coverage:.1f}% cells affected")
    print("- Full generation: ✓ Working correctly")
    print("=" * 60)


if __name__ == "__main__":
    main()
