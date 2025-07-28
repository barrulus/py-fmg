#!/usr/bin/env python3
"""
Test if JavaScript implicitly converts float values when assigning to Uint8Array.
This would explain the 51 vs 52 discrepancy.
"""

import numpy as np

def test_uint8_assignment():
    """Test how values are converted when assigned to Uint8Array."""
    
    print("🔍 UINT8 IMPLICIT CONVERSION TEST")
    print("=" * 50)
    
    # In JavaScript, when you assign a regular array to a Uint8Array,
    # it converts each element. Let's simulate this.
    
    print("JavaScript behavior when assigning to Uint8Array:")
    print("-" * 40)
    
    # Test values that might occur
    test_values = [
        51.0, 51.1, 51.2, 51.3, 51.4, 51.5, 51.6, 51.7, 51.8, 51.9,
        52.0, 52.1, 52.2, 52.3, 52.4, 52.5, 52.6, 52.7, 52.8, 52.9
    ]
    
    print("Value → Uint8 (truncated)")
    for val in test_values:
        # JavaScript Uint8Array truncates (floors) values, not rounds
        uint8_val = int(val)  # This simulates JS behavior
        print(f"  {val:.1f} → {uint8_val}")
    
    print("\n🎯 SOLUTION FOUND:")
    print("-" * 40)
    print("JavaScript Uint8Array truncates (floors) decimal values!")
    print("So 51.9 → 51, 52.0 → 52, 52.9 → 52")
    
    # Test our specific case
    print("\n📊 OUR SPECIFIC CASE:")
    print("-" * 40)
    
    # After multiply, we get 52.0
    # After mask at center, still 52.0
    # But what if there's a slight calculation difference?
    
    center_h = 52.0
    edge_h = 44.85101318359375  # Our observed max
    
    print(f"Center cell: {center_h} → Uint8: {int(center_h)}")
    print(f"Edge cell: {edge_h} → Uint8: {int(edge_h)}")
    
    # What if the FMG calculation resulted in 51.something?
    print("\nIf FMG's max was actually:")
    for h in [51.1, 51.5, 51.9, 51.99, 51.999]:
        print(f"  {h} → Uint8: {int(h)}")
    
    print("\n💡 HYPOTHESIS:")
    print("-" * 40)
    print("The 6-level gap might be due to:")
    print("1. Small floating-point differences in calculations")
    print("2. Different cell being the maximum in FMG")
    print("3. That cell having a value of 51.x instead of 52.0")
    print("4. Uint8Array truncation giving final value of 51")

if __name__ == "__main__":
    test_uint8_assignment()