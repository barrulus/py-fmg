#!/usr/bin/env python3
"""
Compare our packing results with FMG reference data.
"""

import json
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

def compare_packing():
    """Compare our packing with FMG reference."""
    
    print("🔍 PACKING COMPARISON")
    print("=" * 50)
    
    # Load FMG reference data
    with open('/home/user/py-fmg/tests/Mataeu3 Full 2025-07-28-06-23.json', 'r') as f:
        fmg_data = json.load(f)
    
    fmg_cells = fmg_data['pack']['cells']
    fmg_heights = [cell['h'] for cell in fmg_cells]
    fmg_land_cells = sum(1 for h in fmg_heights if h >= 20)
    
    print("📊 FMG Reference (seed 651658815, lowIsland):")
    print(f"   Total packed cells: {len(fmg_cells)}")
    print(f"   Land cells (h>=20): {fmg_land_cells}")
    print(f"   Height range: {min(fmg_heights)}-{max(fmg_heights)}")
    print(f"   Mean height: {np.mean(fmg_heights):.1f}")
    print()
    
    # Generate our version
    config = GridConfig(width=300, height=300, cells_desired=10000)
    voronoi_graph = generate_voronoi_graph(config, "651658815")
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=10000
    )
    
    generator = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = generator.from_template("lowIsland", "651658815")
    
    print("📊 Our Results (before packing):")
    print(f"   Total points: {len(heights)}")
    print(f"   Land cells (h>=20): {np.sum(heights >= 20)}")
    print(f"   Water cells (h<20): {np.sum(heights < 20)}")
    print(f"   Height range: {np.min(heights)}-{np.max(heights)}")
    print(f"   Mean height: {np.mean(heights):.1f}")
    print()
    
    # Analyze height distribution
    print("🔍 Height Distribution Comparison:")
    print("-" * 30)
    
    height_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
    
    print("Range    | FMG  | Ours")
    print("---------|------|-----")
    for min_h, max_h in height_ranges:
        fmg_count = sum(1 for h in fmg_heights if min_h <= h < max_h)
        our_count = np.sum((heights >= min_h) & (heights < max_h))
        print(f"{min_h:2d}-{max_h:2d}    | {fmg_count:4d} | {our_count:4d}")
    
    print()
    print("🎯 Key Differences:")
    print(f"   FMG keeps {len(fmg_cells)} cells after packing")
    print(f"   We would keep {np.sum(heights >= 20)} cells (h>=20)")
    print(f"   Difference: {len(fmg_cells) - np.sum(heights >= 20):+d} cells")
    
    # Check if FMG uses different threshold
    for threshold in [19, 18, 17, 16, 15]:
        count = np.sum(heights >= threshold)
        if abs(count - len(fmg_cells)) < 50:  # Within 50 cells
            print(f"   📍 If we used threshold {threshold}: {count} cells (diff: {count - len(fmg_cells):+d})")

if __name__ == "__main__":
    compare_packing()