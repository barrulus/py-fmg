#!/usr/bin/env python3
"""
Debug the multiply command specifically to understand why it's destroying terrain.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def debug_multiply_command():
    """Debug the multiply command specifically."""
    
    print("🔍 MULTIPLY COMMAND DEBUG")
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
    
    # Execute template up to the multiply command
    generator.add_hill("1", "90-99", "60-80", "45-55")
    generator.add_hill("1-2", "20-30", "10-30", "10-90")
    generator.smooth("2", "0", "0", "0")
    generator.add_hill("6-7", "25-35", "20-70", "30-70")
    generator.add_range("1", "40-50", "45-55", "45-55")
    generator.add_trough("2-3", "20-30", "15-85", "20-30")
    generator.add_trough("2-3", "20-30", "15-85", "70-80")
    generator.add_hill("1.5", "10-15", "5-15", "20-80")
    generator.add_hill("1", "10-15", "85-95", "70-80")
    generator.add_pit("5-7", "15-25", "15-85", "20-80")
    
    # State before multiply
    pre_heights = generator.heights.copy()
    pre_stats = {
        'range': (np.min(pre_heights), np.max(pre_heights)),
        'mean': np.mean(pre_heights),
        'land_cells': np.sum(pre_heights >= 20),
        'high_terrain': np.sum(pre_heights >= 40)
    }
    
    print("🎯 BEFORE MULTIPLY:")
    print(f"   Range: {pre_stats['range'][0]}-{pre_stats['range'][1]}")
    print(f"   Mean: {pre_stats['mean']:.1f}")
    print(f"   Land cells: {pre_stats['land_cells']}")
    print(f"   High terrain: {pre_stats['high_terrain']}")
    
    # Show height distribution before
    heights_above_40 = pre_heights[pre_heights >= 40]
    print(f"   Heights ≥40: count={len(heights_above_40)}")
    if len(heights_above_40) > 0:
        print(f"   Heights ≥40: min={np.min(heights_above_40)}, max={np.max(heights_above_40)}")
        
    # Sample some specific high values
    high_indices = np.where(pre_heights >= 50)[0][:5]
    print(f"   Sample high values (≥50): {[pre_heights[i] for i in high_indices]}")
    print()
    
    # Execute the multiply command
    print("🔧 EXECUTING: Multiply 0.4 20-100")
    print("   Command interpretation: modify('20-100', multiply=0.4)")
    print("   Expected formula: (h - 20) * 0.4 + 20 for h ≥ 20")
    print()
    
    # Manual calculation examples
    test_heights = [20, 30, 40, 50, 60]
    print("   Manual calculation examples:")
    for h in test_heights:
        expected = (h - 20) * 0.4 + 20
        print(f"     Height {h}: ({h} - 20) * 0.4 + 20 = {expected}")
    print()
    
    generator.modify("20-100", multiply=0.4)
    
    # State after multiply
    post_heights = generator.heights.copy()
    post_stats = {
        'range': (np.min(post_heights), np.max(post_heights)),
        'mean': np.mean(post_heights),
        'land_cells': np.sum(post_heights >= 20),
        'high_terrain': np.sum(post_heights >= 40)
    }
    
    print("🎯 AFTER MULTIPLY:")
    print(f"   Range: {post_stats['range'][0]}-{post_stats['range'][1]}")
    print(f"   Mean: {post_stats['mean']:.1f}")
    print(f"   Land cells: {post_stats['land_cells']}")
    print(f"   High terrain: {post_stats['high_terrain']}")
    
    # Check what happened to our sample high values
    print(f"   Sample values after multiply: {[post_heights[i] for i in high_indices]}")
    
    # Verify the calculation manually
    print()
    print("🔍 MANUAL VERIFICATION:")
    for i in high_indices:
        pre_val = pre_heights[i]
        post_val = post_heights[i]
        expected = (pre_val - 20) * 0.4 + 20
        print(f"   Index {i}: {pre_val} → {post_val} (expected: {expected:.1f})")
    
    print()
    print("📊 IMPACT ANALYSIS:")
    print(f"   Range change: {pre_stats['range'][0]}-{pre_stats['range'][1]} → {post_stats['range'][0]}-{post_stats['range'][1]}")
    print(f"   Mean change: {pre_stats['mean']:.1f} → {post_stats['mean']:.1f} (Δ{post_stats['mean'] - pre_stats['mean']:+.1f})")
    print(f"   High terrain change: {pre_stats['high_terrain']} → {post_stats['high_terrain']} (Δ{post_stats['high_terrain'] - pre_stats['high_terrain']:+d})")
    
    # Check if the issue is with range parsing
    print()
    print("🔍 RANGE PARSING CHECK:")
    print(f"   Range '20-100' should affect cells with heights 20-100")
    affected_cells = np.sum((pre_heights >= 20) & (pre_heights <= 100))
    print(f"   Cells in range 20-100: {affected_cells}")
    
    return pre_heights, post_heights

if __name__ == "__main__":
    debug_multiply_command()