#!/usr/bin/env python3
"""
Trace where the maximum height is located throughout the generation process.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def trace_max_location():
    """Track the location of maximum height through generation."""
    
    print("🔍 MAXIMUM HEIGHT LOCATION TRACE")
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
    
    # Get template
    template = get_template("lowIsland")
    commands = [line.strip() for line in template.strip().split('\n') if line.strip()]
    
    print("📜 Tracking maximum height location:")
    print("-" * 50)
    
    def get_max_info():
        max_h = np.max(generator.heights)
        max_idx = np.argmax(generator.heights)
        max_x, max_y = generator.graph.points[max_idx]
        # Normalize to [0, 1]
        norm_x = max_x / generator.config.width
        norm_y = max_y / generator.config.height
        return max_h, max_idx, max_x, max_y, norm_x, norm_y
    
    # Track through each command
    for i, cmd in enumerate(commands):
        parts = cmd.strip().split()
        if len(parts) < 2:
            continue
            
        command = parts[0]
        args = parts[1:]
        
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
        
        # Get max info after command
        max_h, max_idx, max_x, max_y, norm_x, norm_y = get_max_info()
        
        print(f"{i+1:2d}. {cmd}")
        print(f"    Max height: {max_h:.1f}")
        print(f"    Location: cell {max_idx}, ({max_x:.1f}, {max_y:.1f})")
        print(f"    Normalized: ({norm_x:.3f}, {norm_y:.3f})")
        
        # Calculate distance from edge for mask
        nx = 2 * norm_x - 1  # Convert to [-1, 1]
        ny = 2 * norm_y - 1
        distance = (1 - nx**2) * (1 - ny**2)
        print(f"    Distance factor: {distance:.4f}")
        
        if command == "Mask":
            # Show what mask does to this cell
            power = float(args[0])
            factor = abs(power)
            masked = max_h * distance
            new_h = (max_h * (factor - 1) + masked) / factor
            print(f"    Mask calculation: ({max_h:.1f} * {factor-1} + {masked:.1f}) / {factor} = {new_h:.1f}")
        
        print()
    
    # Final analysis
    print("\n📊 FINAL ANALYSIS:")
    print("-" * 30)
    max_h, max_idx, max_x, max_y, norm_x, norm_y = get_max_info()
    print(f"Final max height: {max_h:.1f}")
    print(f"Location: ({norm_x:.3f}, {norm_y:.3f})")
    
    # Check if it's near an edge
    edge_distance = min(norm_x, 1-norm_x, norm_y, 1-norm_y)
    print(f"Distance from nearest edge: {edge_distance:.3f}")
    
    if edge_distance < 0.15:
        print("⚠️  Maximum is near edge - mask effect is strong!")
    else:
        print("✓ Maximum is not near edge - mask effect is moderate")

if __name__ == "__main__":
    trace_max_location()