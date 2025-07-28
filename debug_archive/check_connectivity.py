#!/usr/bin/env python3
"""
Check grid connectivity to see if that's causing the spread issue.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph


def main():
    # Generate the same grid as in the test
    config = GridConfig(width=300, height=300, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed="1234567")
    
    print(f"Grid dimensions: {graph.cells_x}x{graph.cells_y}")
    print(f"Total cells: {len(graph.points)}")
    print(f"Cell neighbors attribute exists: {hasattr(graph, 'cell_neighbors')}")
    
    if hasattr(graph, 'cell_neighbors'):
        # Check connectivity statistics
        neighbor_counts = [len(neighbors) for neighbors in graph.cell_neighbors]
        
        print(f"\nNeighbor statistics:")
        print(f"  Min neighbors: {min(neighbor_counts)}")
        print(f"  Max neighbors: {max(neighbor_counts)}")
        print(f"  Avg neighbors: {np.mean(neighbor_counts):.2f}")
        
        # Check a specific cell
        cell_1671 = 1671  # The starting cell from our test
        print(f"\nCell {cell_1671}:")
        print(f"  Position: {graph.points[cell_1671]}")
        print(f"  Neighbors: {graph.cell_neighbors[cell_1671]}")
        print(f"  Neighbor count: {len(graph.cell_neighbors[cell_1671])}")
        
        # Check if graph is fully connected
        visited = set()
        queue = [0]  # Start from cell 0
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in graph.cell_neighbors[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        print(f"\nConnectivity check:")
        print(f"  Cells reachable from cell 0: {len(visited)}")
        print(f"  Graph is fully connected: {len(visited) == len(graph.points)}")


if __name__ == "__main__":
    main()