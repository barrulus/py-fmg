#!/usr/bin/env python3
"""
Trace the maximum height through all template commands to see where we lose elevation.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def trace_maximum_height():
    """Execute template step by step, tracking maximum height."""
    
    print("🔍 MAXIMUM HEIGHT TRACE")
    print("=" * 50)
    
    # Setup
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    
    # Get template
    template = get_template("lowIsland")
    commands = [line.strip() for line in template.strip().split('\n') if line.strip()]
    
    print("📜 Executing lowIsland template commands:")
    print("-" * 50)
    
    max_heights = []
    
    for i, cmd in enumerate(commands):
        parts = cmd.strip().split()
        if len(parts) < 2:
            continue
            
        command = parts[0]
        args = parts[1:]
        
        # Get max before command
        before_max = np.max(generator.heights)
        
        # Execute command
        if command == "Hill":
            generator.add_hill(*args)
        elif command == "Pit":
            generator.add_pit(*args)
        elif command == "Range":
            generator.add_range(*args)
        elif command == "Trough":
            generator.add_trough(*args)
        elif command == "Strait":
            generator.add_strait(*args)
        elif command == "Smooth":
            generator.smooth(*args)
        elif command == "Mask":
            generator.mask(float(args[0]))
        elif command == "Add":
            generator.modify(args[1], add=float(args[0]))
        elif command == "Multiply":
            generator.modify(args[1], multiply=float(args[0]))
        
        # Get max after command
        after_max = np.max(generator.heights)
        change = after_max - before_max
        
        max_heights.append(after_max)
        
        # Print result
        print(f"{i+1:2d}. {cmd}")
        print(f"    Max: {before_max:.1f} → {after_max:.1f} ({change:+.1f})")
        
        # Highlight significant drops
        if change < -5:
            print(f"    ⚠️  SIGNIFICANT DROP!")
            
            # For multiply command, show calculation
            if command == "Multiply" and len(args) >= 2:
                mult = float(args[0])
                range_spec = args[1]
                if range_spec.startswith("20-"):
                    # Land-relative calculation
                    expected = int((before_max - 20) * mult + 20)
                    print(f"    Calculation: ({before_max} - 20) * {mult} + 20 = {expected}")
    
    print("\n📊 SUMMARY:")
    print("-" * 30)
    print(f"Starting max: {max_heights[0] if max_heights else 0}")
    print(f"Final max: {max_heights[-1] if max_heights else 0}")
    print(f"Total change: {(max_heights[-1] if max_heights else 0) - (max_heights[0] if max_heights else 0):+.1f}")
    
    # Compare with FMG
    print(f"\n🎯 COMPARISON:")
    print(f"Our final max: {max_heights[-1] if max_heights else 0}")
    print(f"FMG final max: 51")
    print(f"Gap: {51 - (max_heights[-1] if max_heights else 0)} levels missing")
    
    return generator.heights

if __name__ == "__main__":
    trace_maximum_height()