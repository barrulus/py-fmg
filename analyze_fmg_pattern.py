#!/usr/bin/env python3
"""
Analyze the pattern in FMG's grid generation.
"""

import json
import numpy as np

# Load FMG data
with open('voronoi-data-162921633(1).json', 'r') as f:
    fmg_data = json.load(f)

points = np.array(fmg_data['voronoi']['cells']['p'])
print(f"Total points: {len(points)}")

# Analyze Y distribution
y_values = points[:, 1]
print(f"\nY coordinate analysis:")
print(f"Min Y: {np.min(y_values):.2f}")
print(f"Max Y: {np.max(y_values):.2f}")
print(f"Mean Y: {np.mean(y_values):.2f}")

# Check if there's a band pattern
y_bands = {}
band_size = 20  # Check 20-pixel bands
for y in y_values:
    band = int(y // band_size) * band_size
    y_bands[band] = y_bands.get(band, 0) + 1

print(f"\nY distribution by {band_size}-pixel bands:")
for band in sorted(y_bands.keys()):
    if y_bands[band] > 0:
        print(f"  Y {band:3d}-{band+band_size:3d}: {'#' * (y_bands[band] // 20)} ({y_bands[band]} points)")

# Check spacing between adjacent points
print("\nChecking for grid pattern...")
# Sort by Y then X
sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
sorted_points = points[sorted_indices]

# Look at first row (similar Y values)
first_y = sorted_points[0, 1]
first_row = [p for p in sorted_points if abs(p[1] - first_y) < 2]
print(f"Points in first row (Y ≈ {first_y:.1f}): {len(first_row)}")
if len(first_row) > 1:
    x_diffs = [first_row[i][0] - first_row[i-1][0] for i in range(1, min(10, len(first_row)))]
    print(f"X spacing in first row: {[f'{d:.1f}' for d in x_diffs]}")

# Check if it's a subset of a regular grid
print("\nChecking if FMG export is a subset...")
width = 1200
height = 1000
# The Y minimum of 151.5 suggests FMG might be excluding the top ~150 pixels
print(f"Missing Y range: 0 to {np.min(y_values):.1f}")
print(f"Missing Y range: {np.max(y_values):.1f} to {height}")

# Calculate what full grid would have been
full_cells = 10000
spacing = np.sqrt(width * height / full_cells)
expected_rows = int(height / spacing)
expected_cols = int(width / spacing)
print(f"\nExpected full grid: {expected_cols} x {expected_rows} = ~{expected_cols * expected_rows} cells")
print(f"Actual cells: {len(points)}")
print(f"Percentage: {len(points) / (expected_cols * expected_rows) * 100:.1f}%")