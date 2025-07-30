import numpy as np

# Test what happens when we assign h to uint8 array
change = np.zeros(10, dtype=np.uint8)

# Test various height values
test_heights = [10, 30, 50, 90, 100, 150, 200, 255, 300]

for h in test_heights:
    # This is what FMG does: change[start] = h
    change[0] = h
    stored = change[0]
    
    # Calculate what happens after power and multiply
    blob_power = 0.98
    rand_factor = 0.9  # Math.random() * 0.2 + 0.9 ranges from 0.9 to 1.1
    
    next_val = (stored ** blob_power) * rand_factor
    print(f"h={h} -> stored as {stored} -> next cell would get {next_val:.2f}")
    
    # What gets stored in uint8
    change[1] = int(next_val)
    print(f"  -> stored in uint8 as {change[1]}")
    print()