#!/usr/bin/env python3
"""Debug isthmus template execution step by step."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import TEMPLATES
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def debug_template_execution(width=1200, height=1000, cells_desired=10000, template="isthmus", seed="123456"):
    """Debug template execution step by step."""
    
    print(f"\nDebugging {template} template execution")
    print(f"Dimensions: {width}x{height}, Cells: {cells_desired}")
    print(f"Seed: {seed}")
    print("="*60)
    
    # Generate Voronoi graph
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed)
    
    # Create heightmap generator
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Get template commands
    template_str = TEMPLATES[template]
    commands = [cmd.strip() for cmd in template_str.strip().split('\n') if cmd.strip()]
    
    # Execute commands one by one
    for i, command in enumerate(commands):
        print(f"\nStep {i+1}: {command}")
        
        # Parse and execute command
        parts = command.split()
        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        # Execute command manually
        try:
            if cmd_name == "Hill":
                generator.add_hill(*args)
            elif cmd_name == "Pit":
                generator.add_pit(*args)
            elif cmd_name == "Range":
                generator.add_range(*args)
            elif cmd_name == "Trough":
                generator.add_trough(*args)
            elif cmd_name == "Strait":
                generator.add_strait(*args)
            elif cmd_name == "Smooth":
                generator.smooth(*args)
            elif cmd_name == "Mask":
                generator.mask(float(args[0]))
            elif cmd_name == "Add":
                generator.add(float(args[0]))
            elif cmd_name == "Mult":
                generator.mult(float(args[0]))
            elif cmd_name == "Invert":
                generator.invert(*args)
            else:
                print(f"  WARNING: Unknown command '{cmd_name}'")
        except Exception as e:
            print(f"ERROR executing '{command}': {e}")
        
        # Report stats
        heights = generator.heights
        land_cells = np.sum(heights > 20)
        water_cells = np.sum(heights <= 20)
        total_cells = len(heights)
        land_pct = (land_cells / total_cells) * 100
        water_pct = (water_cells / total_cells) * 100
        
        print(f"  Land: {land_cells:,} ({land_pct:.1f}%) | Water: {water_cells:,} ({water_pct:.1f}%)")
        print(f"  Height range: {heights.min()} to {heights.max()}")
        
        # Save intermediate visualization
        save_intermediate_map(generator, graph, f"debug_isthmus_step_{i+1:02d}_{parts[0].lower()}.png", command)
    
    # Final result
    print("\n" + "="*60)
    print("Template execution complete!")
    return generator.heights


def save_intermediate_map(generator, graph, filename, command):
    """Save intermediate heightmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    points = graph.points
    heights = generator.heights
    
    # Use terrain colormap
    scatter = ax.scatter(points[:, 0], points[:, 1], c=heights, 
                        cmap='terrain', s=5, alpha=0.8, vmin=0, vmax=100)
    
    # Add title and info
    land_pct = (np.sum(heights > 20) / len(heights)) * 100
    ax.set_title(f"Command: {command}\nLand: {land_pct:.1f}%", fontsize=12)
    ax.set_xlim(0, generator.config.width)
    ax.set_ylim(0, generator.config.height)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Height')
    
    # Save
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test isthmus with known seed
    debug_template_execution(
        width=1200,
        height=1000,
        cells_desired=10000,
        template="isthmus",
        seed="654321"  # Use map seed for heightmap generation
    )