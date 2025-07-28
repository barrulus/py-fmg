#!/usr/bin/env python3
"""
Detailed debugging of heightmap generation differences.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.utils.random import set_random_seed, get_prng


def main():
    # Load reference data
    with open('tests/Mateau Full 2025-07-27-14-53.json') as f:
        fmg_data = json.load(f)
    
    # Set up parameters
    grid_seed = "1234567"
    map_seed = "651658815"
    width, height = 300, 300
    cells_desired = 10000
    
    # Generate Voronoi graph
    print(f"Generating Voronoi graph with seed '{grid_seed}'...")
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    print(f"Graph dimensions: {graph.cells_x}x{graph.cells_y}, spacing={graph.spacing}")
    print(f"Number of cells: {len(graph.points)}")
    
    # Create heightmap generator
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Set seed and execute template step by step
    set_random_seed(map_seed)
    generator._prng = None
    
    # Get template
    template = get_template("lowIsland")
    lines = template.strip().split('\n')
    
    print("\n\nExecuting lowIsland template step by step:")
    print("=" * 60)
    
    # FMG reference heights
    fmg_heights = np.array([cell['h'] for cell in fmg_data['grid']['cells']])
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        command = parts[0]
        args = parts[1:]
        
        print(f"\nStep {i+1}: {line.strip()}")
        
        # Save state before command
        before_heights = generator.heights.copy()
        
        # Execute command
        if command == "Hill":
            generator.add_hill(*args)
        elif command == "Pit":
            generator.add_pit(*args)
        elif command == "Range":
            generator.add_range(*args)
        elif command == "Trough":
            generator.add_trough(*args)
        elif command == "Strait":
            generator.add_strait(*args)
        elif command == "Smooth":
            generator.smooth(*args)
        elif command == "Mask":
            generator.mask(float(args[0]))
        elif command == "Add":
            generator.modify(args[1], add=float(args[0]))
        elif command == "Multiply":
            generator.modify(args[1], multiply=float(args[0]))
        elif command == "Invert":
            generator.invert(*args)
        
        # Analyze changes
        changed = np.sum(generator.heights != before_heights)
        print(f"  Changed cells: {changed}")
        print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
        print(f"  Land cells (>=20): {np.sum(generator.heights >= 20)} ({np.sum(generator.heights >= 20) / len(generator.heights) * 100:.1f}%)")
        
        # Show some specific values
        if changed > 0:
            # Find first few changed cells
            changed_indices = np.where(generator.heights != before_heights)[0][:5]
            for idx in changed_indices:
                print(f"    Cell {idx}: {before_heights[idx]} -> {generator.heights[idx]}")
    
    print("\n\n" + "=" * 60)
    print("FINAL COMPARISON:")
    print("=" * 60)
    
    print(f"\nPython heights:")
    print(f"  min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  Land: {np.sum(generator.heights >= 20) / len(generator.heights) * 100:.1f}%")
    print(f"  First 20: {generator.heights[:20]}")
    
    print(f"\nFMG heights:")
    print(f"  min={fmg_heights.min()}, max={fmg_heights.max()}, mean={fmg_heights.mean():.1f}")
    print(f"  Land: {np.sum(fmg_heights >= 20) / len(fmg_heights) * 100:.1f}%")
    print(f"  First 20: {fmg_heights[:20]}")
    
    # Exact matches
    matches = generator.heights == fmg_heights
    print(f"\nExact matches: {np.sum(matches)} / {len(generator.heights)} ({np.sum(matches)/len(generator.heights)*100:.1f}%)")


if __name__ == "__main__":
    main()