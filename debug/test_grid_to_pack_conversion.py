#!/usr/bin/env python3
"""
Test the conversion from grid cells to pack cells to see if there's any transformation.
"""

import numpy as np
import json
from pathlib import Path

def analyze_grid_to_pack_conversion():
    """Analyze how grid cell heights become pack cell heights."""
    
    print("🔍 GRID TO PACK CONVERSION ANALYSIS")
    print("=" * 50)
    
    # Load the FMG reference data
    reference_path = Path(__file__).parent.parent / "tests" / "Mataeu3 Full 2025-07-28-06-23.json"
    
    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}")
        return
        
    with open(reference_path, 'r') as f:
        fmg_data = json.load(f)
    
    # Extract grid and pack data
    grid = fmg_data['grid']
    pack = fmg_data['pack']
    
    print(f"Grid cells: {len(grid['cells'])}")
    print(f"Pack cells: {len(pack['cells'])}")
    
    # Get heights from both
    grid_heights = [cell['h'] for cell in grid['cells']]
    pack_heights = [cell['h'] for cell in pack['cells']]
    
    print(f"\nGrid heights range: {min(grid_heights)} - {max(grid_heights)}")
    print(f"Pack heights range: {min(pack_heights)} - {max(pack_heights)}")
    
    # Check the mapping
    print("\n📊 HEIGHT VALUE DISTRIBUTION:")
    print("-" * 30)
    
    # Count unique values
    grid_unique = sorted(set(grid_heights))
    pack_unique = sorted(set(pack_heights))
    
    print(f"Unique grid heights: {len(grid_unique)}")
    print(f"Unique pack heights: {len(pack_unique)}")
    
    # Check if pack heights are a subset of grid heights
    pack_not_in_grid = [h for h in pack_unique if h not in grid_unique]
    if pack_not_in_grid:
        print(f"\n⚠️ Pack heights not in grid: {pack_not_in_grid}")
    else:
        print("\n✓ All pack heights exist in grid heights")
    
    # Look at the transformation
    print("\n🔍 PACK CELL MAPPING:")
    print("-" * 30)
    
    # Sample some pack cells
    for i in range(min(10, len(pack['cells']))):
        pack_cell = pack['cells'][i]
        grid_idx = pack_cell['g']  # grid cell index
        grid_cell = grid['cells'][grid_idx]
        
        print(f"Pack cell {i}: h={pack_cell['h']}, from grid cell {grid_idx} with h={grid_cell['h']}")
        
        if pack_cell['h'] != grid_cell['h']:
            print(f"  ⚠️ HEIGHT MISMATCH!")
    
    # Check for the maximum heights
    print("\n🎯 MAXIMUM HEIGHT ANALYSIS:")
    print("-" * 30)
    
    max_grid_h = max(grid_heights)
    max_pack_h = max(pack_heights)
    
    # Find cells with max height
    max_grid_cells = [i for i, h in enumerate(grid_heights) if h == max_grid_h]
    max_pack_cells = [i for i, h in enumerate(pack_heights) if h == max_pack_h]
    
    print(f"Max grid height {max_grid_h} in {len(max_grid_cells)} cells")
    print(f"Max pack height {max_pack_h} in {len(max_pack_cells)} cells")
    
    # Check if max grid cells made it to pack
    max_grid_in_pack = 0
    for pack_idx, pack_cell in enumerate(pack['cells']):
        if pack_cell['g'] in max_grid_cells:
            max_grid_in_pack += 1
            print(f"  Grid cell {pack_cell['g']} (h={max_grid_h}) → Pack cell {pack_idx} (h={pack_cell['h']})")
    
    print(f"\n{max_grid_in_pack} of {len(max_grid_cells)} max grid cells included in pack")
    
    # Check the packing criteria
    print("\n📦 PACKING CRITERIA (from reGraph):")
    print("-" * 30)
    print("Grid cells included in pack if:")
    print("- height >= 20 (land), OR")
    print("- type == -1 or -2 (coastline/lake)")
    print("- Additional points added for coastal cells")

if __name__ == "__main__":
    analyze_grid_to_pack_conversion()