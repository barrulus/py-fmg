#!/usr/bin/env python3
"""
Debug template execution step-by-step to identify where height distribution goes wrong.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template

def debug_template_step_by_step():
    """Debug each template command execution to track height evolution."""
    
    print("🔍 TEMPLATE EXECUTION STEP-BY-STEP DEBUG")
    print("=" * 60)
    
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
    
    # Get lowIsland template
    template = get_template("lowIsland")
    print(f"📜 Template: lowIsland")
    print(f"   Commands:")
    for line in template.strip().split('\n'):
        print(f"     {line.strip()}")
    print()
    
    # Parse commands (newline-separated, not semicolon)
    lines = template.strip().split('\n')
    commands = [line.strip() for line in lines if line.strip()]
    
    print("🎯 INITIAL STATE:")
    print("-" * 30)
    print(f"   Heights: all zeros")
    print(f"   Range: 0-0")
    print(f"   Mean: 0.0")
    print()
    
    # Execute each command and track changes
    for i, cmd in enumerate(commands):
        print(f"📋 COMMAND {i+1}: {cmd}")
        print("-" * 40)
        
        # Store pre-command state
        pre_heights = generator.heights.copy()
        pre_stats = {
            'range': (np.min(pre_heights), np.max(pre_heights)),
            'mean': np.mean(pre_heights),
            'land_cells': np.sum(pre_heights >= 20),
            'high_terrain': np.sum(pre_heights >= 40)
        }
        
        # Parse and execute the command
        parts = cmd.strip().split()
        if len(parts) < 2:
            print("   ⚠️  Invalid command format")
            continue
            
        command = parts[0]
        args = parts[1:]
        
        print(f"   Type: {command}")
        print(f"   Args: {args}")
        
        try:
            # Execute the command using the generator's built-in methods
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
                # Template format: Add value range unused unused
                generator.modify(args[1], add=float(args[0]))
            elif command == "Multiply":
                # Template format: Multiply value range unused unused  
                generator.modify(args[1], multiply=float(args[0]))
            else:
                print(f"   ⚠️  Unknown command: {command}")
                continue
                
        except Exception as e:
            print(f"   ❌ Error executing command: {e}")
            continue
        
        # Analyze post-command state
        post_heights = generator.heights.copy()
        post_stats = {
            'range': (np.min(post_heights), np.max(post_heights)),
            'mean': np.mean(post_heights),
            'land_cells': np.sum(post_heights >= 20),
            'high_terrain': np.sum(post_heights >= 40)
        }
        
        print(f"   Results:")
        print(f"     Range: {pre_stats['range'][0]}-{pre_stats['range'][1]} → {post_stats['range'][0]}-{post_stats['range'][1]}")
        print(f"     Mean: {pre_stats['mean']:.1f} → {post_stats['mean']:.1f} (Δ{post_stats['mean'] - pre_stats['mean']:+.1f})")
        print(f"     Land cells: {pre_stats['land_cells']} → {post_stats['land_cells']} (Δ{post_stats['land_cells'] - pre_stats['land_cells']:+d})")
        print(f"     High terrain: {pre_stats['high_terrain']} → {post_stats['high_terrain']} (Δ{post_stats['high_terrain'] - pre_stats['high_terrain']:+d})")
        
        # Show height distribution after significant changes
        if post_stats['mean'] - pre_stats['mean'] > 2.0:  # Significant change
            heights_above_zero = post_heights[post_heights > 0]
            if len(heights_above_zero) > 0:
                print(f"     Non-zero heights: min={np.min(heights_above_zero)}, max={np.max(heights_above_zero)}, count={len(heights_above_zero)}")
        
        print()
    
    # Final comparison with FMG
    final_heights = generator.heights
    final_stats = {
        'total_cells': len(final_heights),
        'range': (np.min(final_heights), np.max(final_heights)),
        'mean': np.mean(final_heights),
        'land_cells': np.sum(final_heights >= 20),
        'high_terrain': np.sum(final_heights >= 40)
    }
    
    print("🎯 FINAL COMPARISON:")
    print("-" * 30)
    print(f"Our Results:")
    print(f"   Range: {final_stats['range'][0]}-{final_stats['range'][1]}")
    print(f"   Mean: {final_stats['mean']:.1f}")
    print(f"   Land cells: {final_stats['land_cells']}")
    print(f"   High terrain: {final_stats['high_terrain']}")
    print()
    print(f"FMG Target:")
    print(f"   Range: 2-51")
    print(f"   Mean: 26.7")
    print(f"   Land cells: 3207 (of 4264 packed)")
    print(f"   High terrain: 477")
    print()
    print(f"Key Issues:")
    print(f"   ❌ Missing {51 - final_stats['range'][1]} height levels at top")
    print(f"   ❌ Mean too low by {26.7 - final_stats['mean']:.1f}")
    print(f"   ❌ Missing {477 - final_stats['high_terrain']} high terrain cells")
    
    return final_heights

if __name__ == "__main__":
    debug_template_step_by_step()