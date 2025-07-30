import numpy as np

# Test how numpy handles uint8 overflow
arr = np.zeros(5, dtype=np.uint8)

# Test values that would overflow
test_values = [100, 200, 255, 256, 300, 400, 500]

for val in test_values:
    arr[0] = val
    print(f"Assigned {val}, stored as {arr[0]}")

# Test with float values
print("\nFloat values:")
float_values = [100.5, 255.9, 256.1, 300.7]
for val in float_values:
    arr[0] = val
    print(f"Assigned {val}, stored as {arr[0]}")

# Test JavaScript-like behavior
print("\nJavaScript-like int() then assign:")
for val in float_values:
    int_val = int(val)
    arr[0] = int_val
    print(f"Assigned int({val}) = {int_val}, stored as {arr[0]}")