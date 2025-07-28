#!/usr/bin/env python3
"""
Analyze the Hill command implementation to understand height generation differences.
Focus on the first command that should generate the highest peak.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import get_prng

def analyze_first_hill():
    """Analyze the first Hill command that should create the highest peak."""
    
    print("🔍 FIRST HILL COMMAND ANALYSIS")
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
    
    # Get PRNG state before command
    prng = get_prng()
    print(f"🎲 Initial PRNG state check:")
    test_randoms = [prng.random() for _ in range(5)]
    print(f"   First 5 randoms: {[f'{r:.6f}' for r in test_randoms]}")
    
    # Reset PRNG for actual command
    from py_fmg.utils.random import set_random_seed
    set_random_seed(main_seed)
    
    print("\n📜 Command: Hill 1 90-99 60-80 45-55")
    print("   Interpretation:")
    print("   - Count: 1 hill")
    print("   - Height range: 90-99")
    print("   - X range: 60-80% (180-240 on 300px canvas)")
    print("   - Y range: 45-55% (135-165 on 300px canvas)")
    
    # Execute just the first hill command
    pre_heights = generator.heights.copy()
    generator.add_hill("1", "90-99", "60-80", "45-55")
    post_heights = generator.heights.copy()
    
    # Analyze the result
    changed_cells = np.where(post_heights != pre_heights)[0]
    print(f"\n📊 RESULTS:")
    print(f"   Cells changed: {len(changed_cells)}")
    print(f"   Height range: {np.min(post_heights)}-{np.max(post_heights)}")
    print(f"   Max height location: cell {np.argmax(post_heights)}")
    
    # Find the peak (highest cell)
    peak_idx = np.argmax(post_heights)
    peak_height = post_heights[peak_idx]
    peak_x, peak_y = voronoi_graph.points[peak_idx]
    
    print(f"\n🏔️ PEAK ANALYSIS:")
    print(f"   Peak height: {peak_height}")
    print(f"   Peak location: ({peak_x:.1f}, {peak_y:.1f})")
    print(f"   Expected: height 90-99, location (180-240, 135-165)")
    
    # Check if peak is in expected range
    in_x_range = 180 <= peak_x <= 240
    in_y_range = 135 <= peak_y <= 165
    print(f"   Peak in X range: {'✅' if in_x_range else '❌'}")
    print(f"   Peak in Y range: {'✅' if in_y_range else '❌'}")
    
    # Analyze height distribution around peak
    print(f"\n📈 HEIGHT DISTRIBUTION:")
    height_ranges = [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (40, 50)]
    for min_h, max_h in height_ranges:
        count = np.sum((post_heights >= min_h) & (post_heights < max_h))
        if count > 0:
            print(f"   {min_h:2d}-{max_h:3d}: {count:4d} cells")
    
    # Check initial height assignment
    print(f"\n🔍 INITIAL HEIGHT ASSIGNMENT:")
    print(f"   Expected: Random value between 90-99")
    print(f"   Actual peak height: {peak_height}")
    print(f"   Issue: {'Height seems capped!' if peak_height < 90 else 'Height in range'}")
    
    # Test height generation manually
    print(f"\n🧪 MANUAL HEIGHT GENERATION TEST:")
    set_random_seed(main_seed)
    prng = get_prng()
    
    # Skip randoms used for position selection
    _ = prng.random()  # x position
    _ = prng.random()  # y position
    
    # The height random
    height_random = prng.random()
    expected_height = int(90 + height_random * (99 - 90 + 1))
    print(f"   Height random: {height_random:.6f}")
    print(f"   Expected height: {expected_height}")
    print(f"   Actual peak height: {peak_height}")
    
    return post_heights

if __name__ == "__main__":
    analyze_first_hill()