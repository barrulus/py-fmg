#!/usr/bin/env python3
"""Calculate theoretical cell spread for different heights."""

import math

def calculate_spread_distance(initial_height, blob_power=0.98):
    """Calculate how many steps until height drops below 1."""
    h = initial_height
    distance = 0
    
    while True:
        # Use average multiplier of 1.0
        h = (h ** blob_power) * 1.0
        if int(h) <= 1:
            break
        h = int(h)  # Truncate like uint8
        distance += 1
        if distance > 100:  # Safety
            break
    
    return distance

def estimate_cells_affected(distance, avg_neighbors=6):
    """Estimate total cells affected by spreading to given distance."""
    # This is a rough estimate assuming tree-like spreading
    # In reality, cells overlap so it's less than this
    total = 1  # Starting cell
    
    for d in range(distance):
        # Each layer has roughly neighbors^d cells, but with overlap
        # Use a damping factor
        layer_cells = (avg_neighbors - 1) ** d
        total += layer_cells
    
    return total

print("Height spread analysis:")
print("Height | Distance | Est. Cells")
print("-" * 35)

for h in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    dist = calculate_spread_distance(h)
    cells = estimate_cells_affected(dist, avg_neighbors=5.9)
    print(f"{h:6d} | {dist:8d} | {cells:10.0f}")

# Now check what's happening with our actual case
print("\nFor height=50 (isthmus template):")
h = 50
current = h
print("Step | Height | Next | Int | Continue")
print("-" * 40)

for step in range(25):
    next_h = (current ** 0.98) * 1.0  # Average case
    int_next = int(next_h)
    continues = int_next > 1
    
    print(f"{step:4d} | {current:6.1f} | {next_h:4.1f} | {int_next:3d} | {continues}")
    
    if not continues:
        print(f"\nSpreading stops at step {step}")
        break
        
    current = int_next