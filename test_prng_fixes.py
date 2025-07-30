#!/usr/bin/env python3
"""
Test script to verify PRNG fixes in the Python FMG port.

This script tests:
1. PRNG reseeding at each pipeline stage
2. Blob spreading with float values
3. Consistent random sequences with same seed
"""

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.utils.random import get_prng, set_random_seed


def test_prng_consistency():
    """Test that same seed produces same results."""
    print("Testing PRNG consistency...")
    
    seed = "123456789"
    width, height = 1000, 500
    cells_desired = 5000
    
    # Generate graph twice with same seed
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    
    graph1 = generate_voronoi_graph(config, seed=seed, apply_relaxation=False)
    graph2 = generate_voronoi_graph(config, seed=seed, apply_relaxation=False)
    
    # Compare point positions
    points_match = np.allclose(graph1.points, graph2.points)
    print(f"  Graph points match: {points_match}")
    
    # Generate heightmaps twice with same seed
    hm_config = HeightmapConfig(
        width=width, height=height, 
        cells_x=graph1.cells_x, cells_y=graph1.cells_y,
        cells_desired=cells_desired
    )
    
    hm_gen1 = HeightmapGenerator(hm_config, graph1, seed=seed)
    heights1 = hm_gen1.from_template("volcano", seed=seed)
    
    hm_gen2 = HeightmapGenerator(hm_config, graph2, seed=seed)
    heights2 = hm_gen2.from_template("volcano", seed=seed)
    
    heights_match = np.array_equal(heights1, heights2)
    print(f"  Heightmap values match: {heights_match}")
    
    return points_match and heights_match


def test_blob_spreading():
    """Test that blob spreading works with float values."""
    print("\nTesting blob spreading...")
    
    seed = "123456789"
    width, height = 500, 500
    cells_desired = 1000
    
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed, apply_relaxation=False)
    
    hm_config = HeightmapConfig(
        width=width, height=height,
        cells_x=graph.cells_x, cells_y=graph.cells_y,
        cells_desired=cells_desired
    )
    
    hm_gen = HeightmapGenerator(hm_config, graph, seed=seed)
    
    # Add a single hill and check spreading
    hm_gen.add_hill(count=1, height=50, range_x="45-55", range_y="45-55")
    
    # Check that we have non-zero heights
    non_zero_cells = np.sum(hm_gen.heights > 0)
    print(f"  Non-zero cells after blob spreading: {non_zero_cells}")
    print(f"  Max height: {np.max(hm_gen.heights):.2f}")
    print(f"  Mean height (non-zero): {np.mean(hm_gen.heights[hm_gen.heights > 0]):.2f}")
    
    # Check for float values (not just integers)
    has_float_values = np.any((hm_gen.heights % 1) != 0)
    print(f"  Has fractional height values: {has_float_values}")
    
    return non_zero_cells > 10  # Should spread to more than 10 cells


def test_prng_state_tracking():
    """Test PRNG state at different stages."""
    print("\nTesting PRNG state tracking...")
    
    seed = "123456789"
    
    # Track initial state
    set_random_seed(seed)
    prng1 = get_prng()
    initial_count = prng1.call_count
    print(f"  Initial PRNG call count: {initial_count}")
    
    # Generate some random numbers
    values1 = [prng1.random() for _ in range(5)]
    print(f"  First 5 values: {[f'{v:.6f}' for v in values1]}")
    
    # Reseed and verify we get same sequence
    set_random_seed(seed)
    prng2 = get_prng()
    values2 = [prng2.random() for _ in range(5)]
    print(f"  After reseed: {[f'{v:.6f}' for v in values2]}")
    
    sequences_match = values1 == values2
    print(f"  Sequences match after reseed: {sequences_match}")
    
    return sequences_match


def main():
    """Run all tests."""
    print("Running PRNG fix verification tests...\n")
    
    tests = [
        ("PRNG Consistency", test_prng_consistency),
        ("Blob Spreading", test_blob_spreading),
        ("PRNG State Tracking", test_prng_state_tracking)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)