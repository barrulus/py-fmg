#!/usr/bin/env python3
"""Calculate theoretical spreading depth for different heights."""

import numpy as np

def calculate_max_spread_distance(initial_height, blob_power=0.98):
    """Calculate maximum spreading distance for a given initial height."""
    h = initial_height
    distance = 0
    
    # Track the decay
    print(f"Starting height: {initial_height}")
    print("Distance | Height | Int | Continues?")
    print("-" * 40)
    
    while True:
        int_h = int(h)
        continues = int_h > 1
        
        if distance < 20 or not continues:
            print(f"{distance:8d} | {h:6.2f} | {int_h:3d} | {continues}")
        
        if not continues:
            break
            
        # Calculate next height (worst case: multiply by 1.1)
        h = (int_h ** blob_power) * 1.1
        distance += 1
        
        if distance > 200:
            print("... (stopping at 200)")
            break
    
    return distance

# Test with different starting heights
for start_h in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    max_dist = calculate_max_spread_distance(start_h)
    print(f"\nHeight {start_h} -> max distance: {max_dist}")
    
# Now let's think about the graph structure
print("\n\nGraph connectivity analysis:")
print("-" * 40)

# With ~6 neighbors per cell on average
avg_neighbors = 6

# Calculate approximate cells at each distance
print("Distance | Approx cells (no overlap)")
print("-" * 40)

total_cells = 1
for dist in range(20):
    # Rough approximation - in reality there's significant overlap
    cells_at_dist = avg_neighbors ** dist
    total_cells += cells_at_dist
    print(f"{dist:8d} | {cells_at_dist:12.0f}")
    
    if total_cells > 10000:
        print(f"\nTotal cells would exceed 10,000 at distance {dist}")
        break