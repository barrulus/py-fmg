#!/usr/bin/env python3
"""Test different interpretations of blob power."""

# Test with original blob power
blob_power = 0.98
height = 25

print("Theory 1: Use blob_power directly (current implementation)")
print(f"Height^{blob_power} = {height**blob_power:.2f}")
print(f"With factor 0.9: {height**blob_power * 0.9:.2f}")
print(f"With factor 1.1: {height**blob_power * 1.1:.2f}")
print()

# Test with 1 - blob_power
alt_power = 1 - blob_power
print(f"Theory 2: Use (1 - blob_power) = {alt_power}")
print(f"Height^{alt_power} = {height**alt_power:.2f}")
print(f"With factor 0.9: {height**alt_power * 0.9:.2f}")
print(f"With factor 1.1: {height**alt_power * 1.1:.2f}")
print()

# Test decay over distance with both methods
print("Decay over distance (Theory 1 - current):")
current = height
for i in range(10):
    current = current ** blob_power * 1.0  # Average factor
    print(f"  Distance {i+1}: {current:.2f}")

print("\nDecay over distance (Theory 2 - inverted):")
current = height
for i in range(10):
    current = current ** alt_power * 1.0  # Average factor
    print(f"  Distance {i+1}: {current:.2f}")
    if current < 1:
        break

# Check what decay rate would make sense
print("\nWhat power gives reasonable decay?")
for test_power in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    result = height ** test_power
    print(f"  25^{test_power} = {result:.2f}")