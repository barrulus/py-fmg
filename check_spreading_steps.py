#!/usr/bin/env python3
"""Check how many steps blob spreading should take."""

# Starting from height 17 (from our test)
h = 17
blob_power = 0.98
steps = 0

print("Blob spreading decay:")
print("Step | Height | Continue?")
print("-" * 30)

while h > 1 and steps < 200:
    print(f"{steps:4d} | {h:6.2f} | {'Yes' if h > 1 else 'No'}")
    
    # Calculate next height (average case)
    h = (h ** blob_power) * 1.0
    h = int(h)  # Truncate like uint8
    steps += 1
    
    if h <= 1:
        print(f"{steps:4d} | {h:6.2f} | No")
        break

print(f"\nSpreading stops after {steps} steps")
print(f"Our test showed max_dist of 127-148 steps!")
print("\nThis means the spreading is continuing FAR longer than it should!")

# Let's check if there's an issue with how we check new_height > 1
print("\n\nChecking the > 1 condition:")
for val in [2.0, 1.9, 1.5, 1.1, 1.0, 0.9]:
    int_val = int(val)
    continues = int_val > 1
    print(f"  {val} -> int({val}) = {int_val} -> continues: {continues}")
    
print("\nWait! The issue might be that we're checking the float value > 1")
print("but FMG might be checking the truncated value!")

# Test this theory
print("\n\nFMG-style check (if (change[c] > 1)):")
for val in [2.0, 1.9, 1.5, 1.1, 1.0, 0.9]:
    # In FMG, change[c] would already be truncated to uint8
    truncated = int(val)
    continues = truncated > 1
    print(f"  {val} -> uint8 stores {truncated} -> continues: {continues}")