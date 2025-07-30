#!/usr/bin/env python3
"""Debug blob decay to understand the spreading issue."""

# Test blob decay with blob_power = 0.98
blob_power = 0.98
height = 25  # Starting height

print(f"Starting height: {height}")
print(f"Blob power: {blob_power}")
print("\nHeight decay by distance (using factor 0.9):")

current = height
for dist in range(20):
    # Apply the decay formula with minimum random factor
    current = current ** blob_power * 0.9
    print(f"  Distance {dist+1}: {current:.2f}")
    if current <= 1:
        print(f"  Stops at distance {dist+1}")
        break

print("\nHeight decay by distance (using factor 1.1):")
current = height
for dist in range(20):
    # Apply the decay formula with maximum random factor
    current = current ** blob_power * 1.1
    print(f"  Distance {dist+1}: {current:.2f}")
    if current <= 1:
        print(f"  Stops at distance {dist+1}")
        break

# Compare with direct power calculation
print("\nDirect calculation check:")
print(f"25^0.98 = {25**0.98:.2f}")
print(f"25^0.98 * 0.9 = {25**0.98 * 0.9:.2f}")
print(f"25^0.98 * 1.1 = {25**0.98 * 1.1:.2f}")