#!/usr/bin/env python3
"""Debug the change array values during spreading."""

import sys
sys.path.append('/home/user/py-fmg')

import numpy as np

# Test uint8 assignment behavior
print("Testing uint8 assignment behavior:")
print("-" * 40)

# Test what happens with our spreading calculation
change = np.zeros(10, dtype=np.uint8)
blob_power = 0.98

# Starting height
h = 17  # From the test above

# Simulate spreading
current = h
change[0] = current

print(f"Start: change[0] = {change[0]}")

for i in range(1, 10):
    # Use average multiplier
    rand_factor = 1.0  # Average of (0.9 to 1.1)
    new_val = (current ** blob_power) * rand_factor
    
    print(f"Step {i}: {current} ** {blob_power} * {rand_factor} = {new_val:.3f}")
    
    # This is what we do in the code
    if new_val > 1:
        int_val = int(new_val)
        change[i] = int_val
        print(f"         -> int({new_val:.3f}) = {int_val} -> stored as {change[i]}")
        current = change[i]
    else:
        print(f"         -> STOP (value <= 1)")
        break

print(f"\nFinal change array: {change}")

# Now test with FMG's exact calculation
print("\n\nTesting FMG's exact calculation:")
print("-" * 40)

# FMG does: change[c] = change[q] ** blobPower * (Math.random() * 0.2 + 0.9);
# The key is that change[] is Uint8Array, so values are automatically truncated

change_fmg = np.zeros(10, dtype=np.uint8)
h = 17
change_fmg[0] = h

print(f"Start: change[0] = {change_fmg[0]}")

for i in range(1, 10):
    # Get current value from uint8 array
    current_uint8 = change_fmg[i-1]
    
    # Calculate new value
    new_val = (current_uint8 ** blob_power) * 1.0
    
    print(f"Step {i}: uint8({current_uint8}) ** {blob_power} * 1.0 = {new_val:.3f}")
    
    # Direct assignment to uint8 - NumPy will truncate
    if new_val > 1:
        change_fmg[i] = new_val  # NumPy truncates float to uint8
        print(f"         -> assigned {new_val:.3f} -> stored as {change_fmg[i]}")
    else:
        print(f"         -> STOP (value <= 1)")
        break

print(f"\nFinal change array: {change_fmg}")

# The key question: are we reading from the uint8 array or from a float?
print("\n\nKEY INSIGHT:")
print("In FMG: change[c] = change[q] ** blobPower * random")
print("         where change is Uint8Array")
print("In Python: We do the same, but are we reading the uint8 value correctly?")