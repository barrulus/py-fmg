#!/usr/bin/env python3
"""
Test the specific Hill command that's causing issues.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed


def test_specific_hill():
    """Test the problematic Hill command."""
    
    print("🔍 TESTING SPECIFIC HILL COMMAND")
    print("=" * 50)
    
    # Setup
    seed = "651658815"
    set_random_seed(seed)
    
    config = GridConfig(width=300, height=300, cells_desired=10000)
    voronoi_graph = generate_voronoi_graph(config, seed)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000,
        spacing=voronoi_graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    
    # Start with a simple heightmap (all zeros)
    generator.heights[:] = 0
    
    print("Initial state:")
    print(f"  Min: {np.min(generator.heights):.1f}")
    print(f"  Max: {np.max(generator.heights):.1f}")
    print(f"  Cells = 0: {np.sum(generator.heights == 0)}")
    
    # Add a single hill with the problematic parameters
    print("\nAdding Hill 1 25 60 60 (single hill at center)")
    generator.add_hill(1, 25, "60", "60")
    
    print("\nAfter single centered hill:")
    print(f"  Min: {np.min(generator.heights):.1f}")
    print(f"  Max: {np.max(generator.heights):.1f}")
    print(f"  Cells = 0: {np.sum(generator.heights == 0)}")
    print(f"  Cells > 0: {np.sum(generator.heights > 0)}")
    
    # How far did it spread?
    affected_cells = np.sum(generator.heights > 0)
    print(f"  Affected cells: {affected_cells} / 10000 ({affected_cells/100:.1f}%)")
    
    # Reset and try the actual problematic command
    generator.heights[:] = 0
    
    print("\n" + "="*50)
    print("Testing problematic command: Hill 6-7 25-35 20-70 30-70")
    
    # Execute the command
    generator.add_hill("6-7", "25-35", "20-70", "30-70")
    
    print("\nAfter problematic hill command:")
    print(f"  Min: {np.min(generator.heights):.1f}")
    print(f"  Max: {np.max(generator.heights):.1f}")
    print(f"  Cells = 0: {np.sum(generator.heights == 0)}")
    print(f"  Cells > 0: {np.sum(generator.heights > 0)}")
    
    # Analyze spreading
    print("\n📊 SPREADING ANALYSIS:")
    print("-" * 40)
    
    # Check blob power
    print(f"Blob power: {generator.blob_power}")
    
    # Simulate spreading from height 30
    h = 30.0
    spread_values = []
    for i in range(10):
        h = (h ** generator.blob_power) * 0.9  # Using 0.9 as min random
        spread_values.append(h)
        if h < 1:
            break
    
    print("\nSpread decay (worst case with random=0.9):")
    for i, val in enumerate(spread_values):
        print(f"  Step {i+1}: {val:.2f}")
    
    print(f"\nSteps until < 1: {len(spread_values)}")
    
    # The issue: with blob_power = 0.98, spreading is TOO effective
    # It takes many steps to decay below 1, affecting the entire map
    
    print("\n💡 INSIGHT:")
    print("-" * 40)
    print("With blob_power = 0.98, the spreading is extremely aggressive.")
    print("6-7 hills placed across the map will cover everything!")
    print("")
    print("FMG must be doing something different:")
    print("1. Different blob power calculation?")
    print("2. Different spreading algorithm?")
    print("3. Some cells are blocked from spreading?")


if __name__ == "__main__":
    test_specific_hill()