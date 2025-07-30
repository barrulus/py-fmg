#!/usr/bin/env python3
"""Test with different blob powers to see spreading behavior."""

def test_spreading(height, blob_power, max_dist=20):
    """Simulate spreading with given parameters."""
    print(f"\nHeight={height}, blob_power={blob_power}")
    current = height
    for i in range(max_dist):
        # Use average factor (1.0)
        current = current ** blob_power * 1.0
        if current <= 1:
            print(f"  Stops at distance {i+1}")
            break
        if i < 10:
            print(f"  Distance {i+1}: {current:.2f}")
    if current > 1:
        print(f"  Still {current:.2f} at distance {max_dist}")

# Test with different cell counts and their blob powers
test_cases = [
    (1000, 0.93, "1K cells"),
    (5000, 0.97, "5K cells"),
    (10000, 0.98, "10K cells"),
    (20000, 0.99, "20K cells"),
]

for cells, power, label in test_cases:
    print(f"\n{label} (blob_power={power}):")
    test_spreading(25, power, 150)