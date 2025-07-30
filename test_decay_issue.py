#!/usr/bin/env python3
"""Test the decay issue with blob power."""

# Test the exact calculation
blob_power = 0.98

for h in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    base = h ** blob_power
    min_val = base * 0.9
    max_val = base * 1.1
    
    print(f"h={h:3d}: {h} ** 0.98 = {base:.2f}")
    print(f"       Range: {min_val:.2f} to {max_val:.2f}")
    print(f"       Int range: {int(min_val)} to {int(max_val)}")
    
    # Check if it can stay the same
    if int(max_val) >= h:
        print(f"       WARNING: Can round back to {h} or higher!")
    
    print()

# The problem is clear: with blob_power = 0.98, many values can round back to themselves
# This creates areas where the spreading never stops!

print("\nThis explains why spreading goes on forever!")
print("The solution is that FMG must have a different decay mechanism.")

# Let's check what happens if we always decay by at least 1
print("\n\nWhat if we ensure decay?")
print("-" * 40)

h = 50
for step in range(20):
    # Current FMG calculation
    next_float = (h ** 0.98) * 1.0  # average
    next_int = int(next_float)
    
    # What if we ensure it always decreases?
    next_ensured = min(int(next_float), h - 1)
    
    print(f"Step {step}: {h} -> {next_float:.2f} -> int={next_int}, ensured={next_ensured}")
    
    if next_int <= 1:
        print("Would stop here with current logic")
        break
    
    if next_ensured <= 1:
        print("Would stop here with ensured decay")
        break
        
    h = next_ensured  # Use ensured decay for next iteration