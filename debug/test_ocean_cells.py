#!/usr/bin/env python3
"""
Test why our minimum height is 15 while FMG's is 2.
Check ocean cell handling and initial values.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def test_ocean_cells():
    """Test ocean cell generation and minimum values."""
    
    print("🔍 OCEAN CELL AND MINIMUM HEIGHT TEST")
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
    
    # Check initial state
    print("📊 INITIAL STATE:")
    print(f"Heights dtype: {generator.heights.dtype}")
    print(f"All zeros: {np.all(generator.heights == 0)}")
    print(f"Range: {np.min(generator.heights)} - {np.max(generator.heights)}")
    
    # Execute template commands one by one
    template = get_template("lowIsland")
    commands = [line.strip() for line in template.strip().split('\n') if line.strip()]
    
    print("\n📜 TRACKING MINIMUM HEIGHT:")
    print("-" * 40)
    
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
        
        # Get stats
        min_h = np.min(generator.heights)
        ocean_cells = np.sum(generator.heights < 20)
        zero_cells = np.sum(generator.heights == 0)
        low_cells = np.sum((generator.heights > 0) & (generator.heights < 10))
        
        print(f"{i+1:2d}. {cmd}")
        print(f"    Min: {min_h:.1f}, Ocean (<20): {ocean_cells}, Zeros: {zero_cells}, Low (0-10): {low_cells}")
        
        # Check for sudden changes
        if i == 0 and min_h > 10:
            print("    ⚠️  First command created high minimum!")
    
    # Final analysis
    print("\n🎯 FINAL ANALYSIS:")
    final_heights = np.floor(generator.heights).astype(np.uint8)
    
    print(f"\nFloat heights range: {np.min(generator.heights):.2f} - {np.max(generator.heights):.2f}")
    print(f"Uint8 heights range: {np.min(final_heights)} - {np.max(final_heights)}")
    
    # Distribution of low values
    print("\nLow value distribution:")
    for h in range(0, 20):
        count = np.sum(final_heights == h)
        if count > 0:
            print(f"  Height {h}: {count} cells")
    
    print("\n💡 HYPOTHESIS:")
    print("FMG minimum of 2 vs our 15 suggests:")
    print("1. Different ocean/water generation")
    print("2. Different initial spreading patterns")
    print("3. Mask command affecting edges differently")
    print("4. Integer vs float coordinate calculations")

if __name__ == "__main__":
    test_ocean_cells()