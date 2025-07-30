#!/usr/bin/env python3
"""Test blob power decay to understand spreading limits."""

import numpy as np

# Test blob power decay
blob_power = 0.98  # For 10000 cells

# Starting height
h = 50

# Track decay
print("Blob power decay simulation:")
print("Distance | Height | Int | Continues?")
print("-" * 40)

current_h = h
for distance in range(20):
    # Worst case: multiply by 1.1 (0.9 + 0.2)
    # Best case: multiply by 0.9 
    # Average: multiply by 1.0
    
    # Let's use average case
    next_h = (current_h ** blob_power) * 1.0
    int_h = int(next_h)
    
    continues = int_h > 1
    print(f"{distance:8d} | {next_h:6.2f} | {int_h:3d} | {continues}")
    
    if not continues:
        break
        
    current_h = int_h

print("\nNow test with uint8 overflow:")
print("-" * 40)

# What if we start with a very high value?
for start_h in [100, 150, 200, 255]:
    current_h = start_h
    distance = 0
    print(f"\nStarting with h={start_h}:")
    
    while distance < 10:
        next_h = (current_h ** blob_power) * 1.0
        int_h = int(next_h)
        
        # Simulate uint8 wraparound
        uint8_h = int_h % 256
        
        print(f"  {distance}: {current_h} -> {next_h:.1f} -> {int_h} -> uint8: {uint8_h}")
        
        if uint8_h <= 1:
            print(f"  Stops at distance {distance}")
            break
            
        current_h = uint8_h
        distance += 1

# Check FMG's lim function effect
print("\nEffect of lim() clamping to 0-100:")
print("-" * 40)

test_heights = [50, 90, 100, 110, 150]
for h in test_heights:
    clamped = min(max(h, 0), 100)
    print(f"Height {h} -> lim() -> {clamped}")