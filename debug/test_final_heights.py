#!/usr/bin/env python3
"""
Test the final heights after Uint8 truncation.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def test_final_heights():
    """Test final height values with truncation."""
    
    print("🔍 FINAL HEIGHT VALUES TEST")
    print("=" * 50)
    
    # Setup
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000,
        spacing=voronoi_graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    
    # Generate using template
    heights = generator.from_template("lowIsland", seed=main_seed)
    
    print("📊 FINAL HEIGHT STATISTICS:")
    print("-" * 30)
    print(f"Type: {type(heights)}")
    print(f"Dtype: {heights.dtype}")
    print(f"Range: {np.min(heights)} - {np.max(heights)}")
    print(f"Unique values: {len(np.unique(heights))}")
    
    # Check high values
    high_mask = heights >= 40
    high_count = np.sum(high_mask)
    print(f"\nCells with height >= 40: {high_count}")
    
    if high_count > 0:
        high_values = np.unique(heights[high_mask])
        print(f"High values: {high_values}")
    
    # Compare with FMG
    print(f"\n🎯 COMPARISON:")
    print(f"Our max: {np.max(heights)}")
    print(f"FMG max: 51")
    print(f"Gap: {51 - np.max(heights)} levels")
    
    # Check if any cells would have been 51.x before truncation
    print("\n🔍 CHECKING PRE-TRUNCATION VALUES:")
    
    # Get the float values before truncation
    # We need to temporarily disable truncation
    generator_debug = HeightmapGenerator(heightmap_config, voronoi_graph)
    
    # Execute template but get float values
    template = get_template("lowIsland")
    commands = [line.strip() for line in template.strip().split('\n') if line.strip()]
    
    for cmd in commands:
        parts = cmd.strip().split()
        if len(parts) < 2:
            continue
            
        command = parts[0]
        args = parts[1:]
        
        if command == "Hill":
            generator_debug.add_hill(*args)
        elif command == "Pit":
            generator_debug.add_pit(*args)
        elif command == "Range":
            generator_debug.add_range(*args)
        elif command == "Trough":
            generator_debug.add_trough(*args)
        elif command == "Strait":
            generator_debug.add_strait(*args)
        elif command == "Smooth":
            generator_debug.smooth(*args)
        elif command == "Mask":
            generator_debug.mask(float(args[0]))
        elif command == "Add":
            generator_debug.modify(args[1], add=float(args[0]))
        elif command == "Multiply":
            generator_debug.modify(args[1], multiply=float(args[0]))
    
    float_heights = generator_debug.heights
    max_float = np.max(float_heights)
    max_idx = np.argmax(float_heights)
    
    print(f"\nMax float value: {max_float}")
    print(f"Truncated to: {int(max_float)}")
    
    # Check cells near 51
    near_51 = (float_heights >= 51.0) & (float_heights < 52.0)
    count_near_51 = np.sum(near_51)
    
    print(f"\nCells with 51.0 <= height < 52.0: {count_near_51}")
    if count_near_51 > 0:
        values_near_51 = float_heights[near_51]
        print(f"Sample values: {values_near_51[:10]}")

if __name__ == "__main__":
    test_final_heights()