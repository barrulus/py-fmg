#!/usr/bin/env python3
"""
Analyze the mathematical spreading behavior to understand why FMG gets different results.
"""

import numpy as np


def simulate_blob_spread(initial_height, blob_power, grid_size=100, start_pos=(50, 50)):
    """
    Simulate blob spreading on a grid.
    """
    grid = np.zeros((grid_size, grid_size))
    visited = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Start position
    sx, sy = start_pos
    grid[sx, sy] = initial_height
    visited[sx, sy] = True
    
    # BFS queue
    queue = [(sx, sy, initial_height)]
    cells_affected = 1
    max_distance = 0
    
    while queue:
        x, y, height = queue.pop(0)
        
        # Process neighbors
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < grid_size and 0 <= ny < grid_size and not visited[nx, ny]:
                # Calculate new height with minimum random factor
                new_height = (height ** blob_power) * 0.9  # Min random factor
                
                if new_height > 1:
                    visited[nx, ny] = True
                    grid[nx, ny] = new_height
                    queue.append((nx, ny, new_height))
                    cells_affected += 1
                    
                    # Track max distance
                    dist = abs(nx - sx) + abs(ny - sy)
                    max_distance = max(max_distance, dist)
    
    return grid, cells_affected, max_distance


def analyze_spreading():
    """Analyze spreading behavior with different parameters."""
    
    print("Blob Spreading Analysis")
    print("=" * 60)
    
    # Test different blob powers
    blob_powers = [0.90, 0.93, 0.95, 0.98, 0.99]
    initial_heights = [50, 70, 92]
    
    results = []
    
    for power in blob_powers:
        for height in initial_heights:
            grid, cells, max_dist = simulate_blob_spread(height, power)
            
            # Calculate spread percentage
            total_cells = grid.shape[0] * grid.shape[1]
            spread_pct = cells / total_cells * 100
            
            results.append({
                'power': power,
                'height': height,
                'cells': cells,
                'spread_pct': spread_pct,
                'max_dist': max_dist,
                'max_value': grid.max(),
                'mean_value': grid[grid > 0].mean() if cells > 0 else 0
            })
            
            print(f"\nPower: {power}, Initial Height: {height}")
            print(f"  Cells affected: {cells} ({spread_pct:.1f}%)")
            print(f"  Max distance: {max_dist}")
            print(f"  Max height: {grid.max():.1f}")
            print(f"  Mean height (non-zero): {results[-1]['mean_value']:.1f}")
    
    # Show sample spreading pattern
    print("\n\nSample Spreading Pattern (Power 0.98, Height 92)")
    print("=" * 60)
    grid, cells, _ = simulate_blob_spread(92, 0.98, grid_size=20, start_pos=(10, 10))
    
    # Show center portion of grid
    center = grid[5:15, 5:15]
    print("Center 10x10 portion:")
    for row in center:
        print(" ".join(f"{val:4.0f}" if val > 0 else "   ." for val in row))
    
    # Analyze decay rates
    print("\n\nDecay Analysis")
    print("=" * 60)
    
    for power in [0.93, 0.98]:
        print(f"\nBlob power: {power}")
        value = 92
        steps = 0
        
        while value > 1 and steps < 50:
            value = value ** power * 0.9  # Min random factor
            steps += 1
            if steps <= 5 or value < 5:
                print(f"  Step {steps}: {value:.2f}")
        
        print(f"  Total steps to decay below 1: {steps}")
    
    # Calculate theoretical spread
    print("\n\nTheoretical Maximum Spread")
    print("=" * 60)
    
    for power in [0.93, 0.98]:
        value = 92
        steps = 0
        
        while value > 1:
            value = value ** power * 0.9
            steps += 1
        
        # In a grid, maximum cells at distance d is 4*d (Manhattan distance)
        # Total cells within distance d is approximately 2*d^2
        max_cells = min(2 * steps * steps, 10000)
        
        print(f"\nPower {power}:")
        print(f"  Decay steps: {steps}")
        print(f"  Theoretical max cells: {max_cells} ({max_cells/100:.0f}%)")


if __name__ == "__main__":
    analyze_spreading()