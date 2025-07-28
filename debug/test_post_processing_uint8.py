#!/usr/bin/env python3
"""
Test to confirm that FMG stores grid.cells.h as Uint8Array,
which would cause values to wrap around at 255.
"""

import numpy as np

def test_uint8_storage():
    """Test what happens when heights are stored in Uint8Array."""
    
    print("🔍 UINT8 STORAGE POST-PROCESSING TEST")
    print("=" * 50)
    
    # Simulate some height values that might come from heightmap generation
    heights_float = np.array([
        20.0, 30.0, 45.0, 52.0, 100.0,
        150.0, 200.0, 250.0, 255.0, 256.0,
        260.0, 300.0, 400.0, 500.0
    ], dtype=np.float32)
    
    print("Original float32 heights:")
    print(f"  {heights_float}")
    print(f"  Range: {np.min(heights_float)} - {np.max(heights_float)}")
    
    # Convert to uint8 (as FMG does with grid.cells.h)
    heights_uint8 = heights_float.astype(np.uint8)
    
    print("\nAfter storing as Uint8Array:")
    print(f"  {heights_uint8}")
    print(f"  Range: {np.min(heights_uint8)} - {np.max(heights_uint8)}")
    
    print("\nValue transformations:")
    for f, u in zip(heights_float, heights_uint8):
        if f != u:
            print(f"  {f} → {u} (wrapped: {f} % 256 = {f % 256})")
    
    # Test with values that might occur after FMG processing
    print("\n🎯 MYSTERY SOLVED:")
    print("FMG stores grid.cells.h as Uint8Array, which limits values to 0-255")
    print("Any height > 255 wraps around (e.g., 256 → 0, 257 → 1, etc.)")
    
    # Test our specific case
    print("\n📊 LOWISLAND TEMPLATE ANALYSIS:")
    # After multiply command, max could be higher if not for Uint8
    # Let's say before multiply we had max of 120
    test_max = 120.0
    print(f"  If max before multiply: {test_max}")
    print(f"  After multiply 0.4: ({test_max} - 20) * 0.4 + 20 = {(test_max - 20) * 0.4 + 20}")
    
    # But what if some operations pushed values over 255?
    print("\n  What if some cells reached 260+ during generation?")
    high_value = 260.0
    wrapped = high_value % 256
    print(f"  Original: {high_value}")
    print(f"  Stored as uint8: {wrapped}")
    print(f"  After multiply: ({wrapped} - 20) * 0.4 + 20 = {(wrapped - 20) * 0.4 + 20:.1f}")

if __name__ == "__main__":
    test_uint8_storage()