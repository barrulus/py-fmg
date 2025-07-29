#!/usr/bin/env python3
"""
Demonstration of the new Voronoi generation improvements.

This script shows the key features added to match FMG's behavior:
1. Height pre-allocation
2. Grid reuse logic
3. Lloyd's relaxation
4. State tracking
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, generate_or_reuse_grid


def main():
    # Configuration
    config = GridConfig(width=100, height=100, cells_desired=100)

    print("=== Voronoi Generation Improvements Demo ===\n")

    # 1. Generate initial grid with Lloyd's relaxation
    print("1. Generating initial grid with Lloyd's relaxation...")
    grid = generate_voronoi_graph(config, seed="demo_seed", apply_relaxation=True)
    print(f"   - Generated {len(grid.points)} cells")
    print(f"   - Heights pre-allocated: {grid.heights.shape}")
    print(f"   - Graph dimensions: {grid.graph_width}x{grid.graph_height}")
    print(f"   - Seed: {grid.seed}")

    # 2. Demonstrate height modification (simulating heightmap generation)
    print("\n2. Simulating heightmap generation...")
    # In FMG, the heightmap generator modifies the pre-allocated heights
    grid.heights[:50] = np.random.randint(20, 100, 50)  # Land cells
    grid.heights[50:] = np.random.randint(0, 20, len(grid.points) - 50)  # Ocean cells
    print(f"   - Modified heights: min={grid.heights.min()}, max={grid.heights.max()}")

    # 3. Demonstrate grid reuse
    print("\n3. Testing grid reuse logic...")

    # Same parameters - should reuse
    print("   a) Same config and seed - should reuse:")
    grid2 = generate_or_reuse_grid(grid, config, "demo_seed")
    print(f"      Grid reused: {grid2 is grid}")

    # Different seed - should regenerate
    print("   b) Different seed - should regenerate:")
    grid3 = generate_or_reuse_grid(grid, config, "different_seed")
    print(f"      Grid regenerated: {grid3 is not grid}")
    print(f"      New seed: {grid3.seed}")

    # Different size - should regenerate
    print("   c) Different size - should regenerate:")
    config2 = GridConfig(width=150, height=150, cells_desired=100)
    grid4 = generate_or_reuse_grid(grid, config2, "demo_seed")
    print(f"      Grid regenerated: {grid4 is not grid}")

    # 4. Compare with/without relaxation
    print("\n4. Comparing with and without Lloyd's relaxation...")
    grid_unrelaxed = generate_voronoi_graph(
        config, "compare_seed", apply_relaxation=False
    )
    grid_relaxed = generate_voronoi_graph(config, "compare_seed", apply_relaxation=True)

    # Calculate average nearest neighbor distance as a measure of uniformity
    def avg_nearest_neighbor(points):
        from scipy.spatial import distance_matrix

        dist_matrix = distance_matrix(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = np.min(dist_matrix, axis=1)
        return np.mean(nearest_distances)

    ann_unrelaxed = avg_nearest_neighbor(grid_unrelaxed.points)
    ann_relaxed = avg_nearest_neighbor(grid_relaxed.points)

    print(f"   - Avg nearest neighbor (unrelaxed): {ann_unrelaxed:.2f}")
    print(f"   - Avg nearest neighbor (relaxed): {ann_relaxed:.2f}")
    print(
        f"   - Improvement: {((ann_relaxed - ann_unrelaxed) / ann_unrelaxed * 100):.1f}%"
    )

    # 5. Show mutable nature of the dataclass
    print("\n5. Demonstrating mutable VoronoiGraph...")
    print(f"   - Original heights sum: {grid.heights.sum()}")
    grid.heights += 10  # Modify heights in-place
    print(f"   - Modified heights sum: {grid.heights.sum()}")

    print("\n=== Demo Complete ===")
    print("\nKey improvements implemented:")
    print("✓ Height pre-allocation for FMG compatibility")
    print("✓ Grid reuse logic for interactive workflows")
    print("✓ Lloyd's relaxation for better point distribution")
    print("✓ Mutable dataclass for stateful operations")
    print("✓ Generation parameter tracking")


if __name__ == "__main__":
    main()
