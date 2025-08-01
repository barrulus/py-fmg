#!/usr/bin/env python3
"""
Analyze the FMG point pattern to understand the generation order.
"""

# First few FMG points
fmg_points = [
    [572.81, 106.47],
    [539.19, 115.65],
    [556.51, 113.52],
    [561.99, 110.51],
    [699.45, 119.3]
]

# Analyze the pattern
print("FMG point analysis:")
print("Point coordinates:")
for i, point in enumerate(fmg_points):
    print(f"  {i}: {point}")

print("\nAnalyzing spacing:")
# These points seem to be in the middle of the grid, not starting from top-left
# Let's estimate the grid position
spacing = 10.95
print(f"Expected spacing: {spacing}")

# Calculate approximate grid positions
for i, point in enumerate(fmg_points):
    grid_x = point[0] / spacing
    grid_y = point[1] / spacing
    print(f"  Point {i}: grid position ~({grid_x:.1f}, {grid_y:.1f})")

print("\nThese points appear to be from the middle of the grid, not the beginning.")
print("FMG might be exporting points in a different order than generation order.")