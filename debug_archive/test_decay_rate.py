#!/usr/bin/env python3
"""
Test decay rate with different blob powers.
"""

def test_decay(initial_value, power, rand_factor=0.9):
    """Test how many steps it takes for a value to decay below 1."""
    value = initial_value
    steps = 0
    
    while value > 1 and steps < 1000:
        value = (value ** power) * rand_factor
        steps += 1
    
    return steps


def main():
    # Test with different blob powers
    print("Decay test with initial value 92:")
    print("=" * 50)
    
    blob_powers = {
        100: 0.93,     # 100 cells
        1000: 0.93,    # 1000 cells
        10000: 0.98,   # 10000 cells (our case)
        100000: 0.9973 # 100000 cells
    }
    
    for cells, power in blob_powers.items():
        steps = test_decay(92, power)
        print(f"Cells: {cells:6d}, Blob power: {power:.4f}, Steps to decay: {steps}")
    
    print("\n\nWith initial value 50:")
    print("=" * 50)
    
    for cells, power in blob_powers.items():
        steps = test_decay(50, power)
        print(f"Cells: {cells:6d}, Blob power: {power:.4f}, Steps to decay: {steps}")
    
    # Calculate approximate spread radius
    print("\n\nApproximate spread (assuming square grid):")
    print("=" * 50)
    
    import math
    
    # For 10000 cells (100x100 grid)
    grid_size = 100
    initial_height = 92
    blob_power = 0.98
    
    # Estimate cells affected
    steps = test_decay(initial_height, blob_power)
    # In a BFS, level n has approximately 4n cells (rough approximation)
    total_cells = 1  # Starting cell
    for level in range(1, steps):
        total_cells += min(4 * level, grid_size * grid_size - total_cells)
        if total_cells >= grid_size * grid_size:
            break
    
    print(f"Grid: {grid_size}x{grid_size}")
    print(f"Initial height: {initial_height}")
    print(f"Blob power: {blob_power}")
    print(f"Decay steps: {steps}")
    print(f"Estimated cells affected: {total_cells} ({total_cells/(grid_size*grid_size)*100:.1f}%)")


if __name__ == "__main__":
    main()