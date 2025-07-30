#!/usr/bin/env python3
"""Analyze expected FMG spreading behavior."""

# If FMG uses ~320 PRNG calls for the entire isthmus template:
# - 5 Hill commands with "5-10" count each
# - 5 Trough commands with "4-8" count each
# - 1 Invert command

# Let's work backwards from 320 calls

total_calls = 320

# Subtract calls for parsing ranges
# Each Hill command: 1 call for count + average 7.5 calls for heights
hill_setup_calls = 5 * (1 + 7.5)  # 42.5
# Each Trough command: 1 call for count + average 6 calls for heights  
trough_setup_calls = 5 * (1 + 6)  # 35
# Invert: 1 call
invert_calls = 1

setup_calls = hill_setup_calls + trough_setup_calls + invert_calls
print(f"Setup calls (parsing ranges): ~{setup_calls:.0f}")

remaining_calls = total_calls - setup_calls
print(f"Remaining for actual spreading: ~{remaining_calls:.0f}")

# Average hills per command: 7.5
# Average troughs per command: 6
total_features = 5 * 7.5 + 5 * 6  # 67.5

calls_per_feature = remaining_calls / total_features
print(f"\nAverage PRNG calls per hill/trough: ~{calls_per_feature:.1f}")

# If each cell visited uses 1 PRNG call (for the random factor)
# Then each hill/trough affects about 3 cells on average!

print("\nThis suggests each hill in FMG:")
print(f"- Uses ~{calls_per_feature:.0f} PRNG calls")
print(f"- Affects ~{calls_per_feature:.0f} cells")
print("- Creates a very small, localized bump")

print("\nBut we're seeing:")
print("- 18,000+ PRNG calls per hill")
print("- 9,788 cells affected per hill")
print("- Nearly the entire map covered")

print("\nPossible explanations:")
print("1. FMG has additional early termination logic")
print("2. The grid connectivity is different") 
print("3. The spreading parameters are different")
print("4. There's a bug in how cells are selected or connected")

# Let's calculate what spreading pattern would give us ~3 cells
print("\n\nSpreading analysis:")
print("-" * 50)

# With power 0.98 and starting height 20:
h = 20
cells = 1
for i in range(10):
    h_next = h ** 0.98
    if h_next < 2:
        break
    print(f"Step {i}: height {h:.1f} -> {h_next:.1f}")
    # Assume 6 neighbors, but many are already visited
    new_cells = max(1, 6 - i)  # Fewer new cells as we go out
    cells += new_cells
    h = h_next

print(f"\nWith limited spreading: ~{cells} cells")

# The key insight: FMG must be doing something to limit spreading!
print("\n\nKey insight: FMG must have additional logic to limit spreading!")
print("Possibilities:")
print("1. Range constraints during spreading (we removed these)")
print("2. Different neighbor connectivity") 
print("3. Early termination based on distance from start")
print("4. Different random factor calculation")