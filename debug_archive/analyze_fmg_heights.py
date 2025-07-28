#!/usr/bin/env python3
"""
Analyze the FMG heights data to understand the actual output.
"""

import json
import numpy as np


def main():
    # Load the FMG browser data
    with open('heights.json') as f:
        data = json.load(f)
    
    heights = np.array(data['heights']['data'])
    
    print("FMG Browser Heights Analysis")
    print("=" * 60)
    
    print(f"\nBasic Info:")
    print(f"  Seed: {data['seed']}")
    print(f"  Dimensions: {data['dimensions']['width']}x{data['dimensions']['height']}")
    print(f"  Cells: {data['cells']['actual']} (grid: {data['cells']['x']}x{data['cells']['y']})")
    
    print(f"\nHeight Statistics:")
    print(f"  Min: {data['heights']['min']}")
    print(f"  Max: {data['heights']['max']}")
    print(f"  Mean: {data['heights']['mean']:.1f}")
    print(f"  Land cells: {data['heights']['landCells']} ({data['heights']['landCells'] / len(heights) * 100:.1f}%)")
    print(f"  Water cells: {data['heights']['waterCells']} ({data['heights']['waterCells'] / len(heights) * 100:.1f}%)")
    
    print(f"\nFirst 20 heights: {data['heights']['first20']}")
    
    # Analyze height distribution
    print(f"\n\nHeight Distribution:")
    print("Range   | Count  | Percentage")
    print("--------|--------|----------")
    for h in range(0, 100, 10):
        count = np.sum((heights >= h) & (heights < h+10))
        pct = count / len(heights) * 100
        print(f"{h:2d}-{h+9:2d}  | {count:6d} | {pct:5.1f}%")
    
    # Compare with our Python output
    print(f"\n\nComparison with Python Implementation:")
    print("=" * 60)
    print("                  | FMG    | Python")
    print("------------------|--------|--------")
    print(f"Land percentage   | {data['heights']['landCells'] / len(heights) * 100:5.1f}% | 82.4%")
    print(f"Mean height       | {data['heights']['mean']:5.1f}  | 28.6")
    print(f"Min height        | {data['heights']['min']:5d}  | 6")
    print(f"Max height        | {data['heights']['max']:5d}  | 51")
    
    # Check for patterns
    print(f"\n\nSpatial Pattern Check:")
    # Reshape to 2D grid
    grid_x = data['cells']['x']
    grid_y = data['cells']['y']
    heights_2d = heights.reshape(grid_y, grid_x)
    
    # Check corners
    print(f"Corner values:")
    print(f"  Top-left: {heights_2d[0, 0]}")
    print(f"  Top-right: {heights_2d[0, -1]}")
    print(f"  Bottom-left: {heights_2d[-1, 0]}")
    print(f"  Bottom-right: {heights_2d[-1, -1]}")
    
    # Check center
    center_y, center_x = grid_y // 2, grid_x // 2
    print(f"\nCenter region ({center_x-2}:{center_x+3}, {center_y-2}:{center_y+3}):")
    print(heights_2d[center_y-2:center_y+3, center_x-2:center_x+3])
    
    # Save first 100 heights for detailed comparison
    print(f"\n\nFirst 100 heights for detailed comparison:")
    print(heights[:100].tolist())


if __name__ == "__main__":
    main()