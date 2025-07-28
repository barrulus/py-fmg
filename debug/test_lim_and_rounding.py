#!/usr/bin/env python3
"""
Test lim function and rounding behavior differences.
"""

import numpy as np
import math

def test_lim_behavior():
    """Test the lim function behavior."""
    
    print("🔍 LIM FUNCTION AND ROUNDING TEST")
    print("=" * 50)
    
    # Test values near boundaries
    test_values = [
        -1, -0.5, -0.1, 0, 0.1, 0.5,
        99.4, 99.5, 99.6, 99.9, 99.99, 100,
        100.1, 100.5, 100.9, 101, 105, 150
    ]
    
    print("Testing lim function (clamp to [0, 100]):")
    print("-" * 40)
    
    for val in test_values:
        # Python implementation
        py_result = max(0, min(100, val))
        
        # JavaScript would do the same with minmax
        # But let's check if there's any rounding difference
        print(f"  {val:6.2f} → {py_result:6.2f}")
    
    # Test specific calculations from heightmap
    print("\n📊 HEIGHTMAP CALCULATION ROUNDING:")
    print("-" * 40)
    
    # Test the multiply operation
    h = 100.0
    mult = 0.4
    
    # Land-relative multiply: (h - 20) * mult + 20
    result = (h - 20) * mult + 20
    print(f"\nMultiply operation: ({h} - 20) * {mult} + 20")
    print(f"  Float result: {result}")
    print(f"  After lim: {max(0, min(100, result))}")
    
    # Test with slightly different values
    print("\nTesting near-boundary values:")
    for h in [99.9, 99.99, 100.0, 100.01, 100.1]:
        result = (h - 20) * mult + 20
        limmed = max(0, min(100, result))
        print(f"  h={h:6.2f}: ({h} - 20) * {mult} + 20 = {result:6.4f} → {limmed:6.2f}")
    
    # Test mask calculation with high precision
    print("\n🎯 MASK CALCULATION PRECISION:")
    print("-" * 40)
    
    h = 52.0
    power = 4
    factor = abs(power)
    
    # Test different positions
    positions = [
        (0.5, 0.5),     # center
        (0.86, 0.5),    # near edge
        (0.867, 0.5),   # specific position
        (0.87, 0.5),    # slightly further
    ]
    
    for x_norm, y_norm in positions:
        nx = 2 * x_norm - 1
        ny = 2 * y_norm - 1
        distance = (1 - nx**2) * (1 - ny**2)
        masked = h * distance
        new_h = (h * (factor - 1) + masked) / factor
        
        print(f"\nPosition ({x_norm}, {y_norm}):")
        print(f"  nx={nx:.6f}, ny={ny:.6f}")
        print(f"  distance = {distance:.10f}")
        print(f"  new_h = ({h} * 3 + {masked:.6f}) / 4 = {new_h:.10f}")
        print(f"  rounded = {round(new_h, 2)}")
    
    # Check if 51 vs 52 could be a rounding issue
    print("\n🔍 51 vs 52 ANALYSIS:")
    print("-" * 40)
    
    # What would give us 51 after mask?
    # new_h = (h * 3 + h * distance) / 4 = h * (3 + distance) / 4
    # 51 = h * (3 + distance) / 4
    # h = 51 * 4 / (3 + distance)
    
    target = 51
    for distance in [0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]:
        h_needed = target * 4 / (3 + distance)
        print(f"  To get {target} with distance={distance:.2f}, need h={h_needed:.2f}")

if __name__ == "__main__":
    test_lim_behavior()