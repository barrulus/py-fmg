#!/usr/bin/env python3
"""
Debug template execution step by step to identify excess land generation.
"""

import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def debug_template_execution():
    """Debug each template command to identify where excess land is generated."""
    
    print("🔍 TEMPLATE EXECUTION DEBUGGING")
    print("=" * 60)
    
    # Use exact FMG parameters
    main_seed = "651658815"
    config = GridConfig(width=300, height=300, cells_desired=10000)
    
    # Generate base components
    voronoi_graph = generate_voronoi_graph(config, main_seed)
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    
    # Get the lowIsland template
    template = get_template("lowIsland")
    print(f"📜 Template: {template['name']}")
    print(f"   Commands: {template['template']}")
    print()
    
    # Reset heightmap for fresh analysis
    generator.reset()
    generator.set_seed(main_seed)
    
    # Parse and execute each command individually
    commands = template["template"].strip().split(';')
    
    print("🎯 COMMAND-BY-COMMAND EXECUTION:")
    print("-" * 50)
    
    for i, cmd in enumerate(commands):
        cmd = cmd.strip()
        if not cmd:
            continue
            
        print(f"Command {i+1}: {cmd}")
        
        # Store pre-command state
        pre_heights = generator.heights.copy()
        pre_land = np.sum(pre_heights >= 20)
        
        # Execute command
        if cmd.startswith('add'):
            parts = cmd.split()
            blob_count = int(parts[1])
            range_str = parts[2] if len(parts) > 2 else "0-100"
            
            print(f"   Adding {blob_count} blobs in range {range_str}")
            generator._execute_add_command(blob_count, range_str)
            
        elif cmd.startswith('smooth'):
            parts = cmd.split()
            iterations = int(parts[1]) if len(parts) > 1 else 1
            print(f"   Smoothing {iterations} iterations")
            generator._execute_smooth_command(iterations)
            
        # Analyze post-command state
        post_heights = generator.heights.copy()
        post_land = np.sum(post_heights >= 20)
        land_change = post_land - pre_land
        
        print(f"   Land cells: {pre_land} → {post_land} (Δ{land_change:+d})")
        print(f"   Height range: {np.min(post_heights)}-{np.max(post_heights)}")
        print(f"   Mean height: {np.mean(post_heights):.1f}")
        print()
    
    final_heights = generator.heights
    final_land = np.sum(final_heights >= 20)
    
    print("🎯 FINAL RESULTS:")
    print("-" * 30)
    print(f"   Total land cells: {final_land}")
    print(f"   Expected (FMG): 3531")
    print(f"   Excess: {final_land - 3531}")
    print(f"   Excess percentage: {((final_land - 3531) / 3531 * 100):+.1f}%")
    
    return final_heights

if __name__ == "__main__":
    debug_template_execution()