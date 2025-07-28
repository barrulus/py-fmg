#!/usr/bin/env python3
"""
Test uint8 overflow behavior in our heightmap operations.
"""

import numpy as np

def test_uint8_operations():
    """Test how uint8 operations behave with our calculations."""
    
    print("🔍 UINT8 OVERFLOW TESTING")
    print("=" * 50)
    
    # Test 1: Multiplication that should exceed 255
    print("\n1. Multiplication overflow test:")
    h = np.array([100, 150, 200], dtype=np.uint8)
    print(f"   Original: {h}")
    
    # Direct multiplication
    result1 = h * 2
    print(f"   h * 2: {result1} (dtype: {result1.dtype})")
    
    # Test 2: Our modify operation for land
    print("\n2. Land-relative multiplication (as in modify):")
    h = np.array([60, 80, 100], dtype=np.uint8)
    print(f"   Original: {h}")
    
    # Simulate: (h - 20) * 0.4 + 20
    # This is what happens in our modify function
    result2 = (h - 20) * 0.4 + 20
    print(f"   (h - 20) * 0.4 + 20: {result2} (dtype: {result2.dtype})")
    
    # What happens if we force it to uint8
    result2_uint8 = ((h - 20) * 0.4 + 20).astype(np.uint8)
    print(f"   Forced to uint8: {result2_uint8}")
    
    # Test 3: Addition that causes overflow
    print("\n3. Addition overflow test:")
    h = np.array([200, 250], dtype=np.uint8)
    print(f"   Original: {h}")
    
    result3 = h + 100
    print(f"   h + 100: {result3} (dtype: {result3.dtype})")
    
    # Test 4: Intermediate calculations
    print("\n4. Intermediate calculation test:")
    h = np.uint8(100)
    print(f"   Original h: {h}")
    
    # Blob spreading formula
    power = 0.98
    random_factor = 0.95
    
    # Direct calculation
    result4 = h ** power * random_factor
    print(f"   h ** {power} * {random_factor}: {result4:.2f} (type: {type(result4)})")
    
    # If we store back to uint8
    result4_uint8 = np.uint8(result4)
    print(f"   Stored as uint8: {result4_uint8}")
    
    # Test 5: Smooth operation behavior
    print("\n5. Smooth operation test:")
    h = np.array([10, 100, 200, 50], dtype=np.uint8)
    print(f"   Original: {h}")
    
    # Averaging neighbors (simplified smooth)
    avg = (h[0] + h[1] + h[2] + h[3]) / 4
    print(f"   Average: {avg} (type: {type(avg)})")
    
    # What if sum overflows?
    sum_vals = h[0] + h[1] + h[2] + h[3]
    print(f"   Sum: {sum_vals} (dtype: {sum_vals.dtype if hasattr(sum_vals, 'dtype') else type(sum_vals)})")

if __name__ == "__main__":
    test_uint8_operations()