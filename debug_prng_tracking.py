#!/usr/bin/env python3
"""Debug script to track PRNG calls during heightmap generation."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import TEMPLATES
from py_fmg.utils.random import get_prng, set_random_seed
import numpy as np

def debug_prng_tracking(width=1200, height=1000, cells_desired=10000, template="isthmus", seed="654321"):
    """Debug PRNG call tracking for template execution."""
    
    print(f"\nPRNG Call Tracking for {template} template")
    print(f"Dimensions: {width}x{height}, Cells: {cells_desired}")
    print(f"Seed: {seed}")
    print("="*60)
    
    # Set seed and get PRNG reference
    set_random_seed(seed)
    prng = get_prng()
    
    # Generate Voronoi graph
    print("\nGenerating Voronoi graph...")
    initial_count = prng.call_count
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=seed)
    print(f"PRNG calls for Voronoi generation: {prng.call_count - initial_count}")
    
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
    
    print(f"\nTotal PRNG calls before template execution: {prng.call_count}")
    print("\nExecuting template commands:")
    print("-"*60)
    
    # Execute commands one by one
    for i, command in enumerate(commands):
        # Record PRNG count before command
        count_before = prng.call_count
        
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
            elif cmd_name == "Mult" or cmd_name == "Multiply":
                generator.mult(float(args[0]))
            elif cmd_name == "Invert":
                generator.invert(*args)
            else:
                print(f"  WARNING: Unknown command '{cmd_name}'")
        except Exception as e:
            print(f"ERROR executing '{command}': {e}")
        
        # Report PRNG calls for this command
        calls_for_command = prng.call_count - count_before
        print(f"Step {i+1}: {command}")
        print(f"  PRNG calls: {calls_for_command}")
        print(f"  Total calls so far: {prng.call_count}")
        
        # Also report land/water stats
        heights = generator.heights
        land_cells = np.sum(heights > 20)
        total_cells = len(heights)
        land_pct = (land_cells / total_cells) * 100
        print(f"  Land: {land_pct:.1f}%")
    
    print("-"*60)
    print(f"\nTotal PRNG calls for entire process: {prng.call_count}")
    
    # For FMG comparison, create JavaScript snippet
    print("\n\nFor FMG JavaScript comparison, add this to heightmap-generator.js:")
    print("""
let random_call_count = 0;
const original_random = Math.random;
Math.random = function() { 
    random_call_count++; 
    return original_random.call(this);
}

// In fromTemplate function, after each command:
console.log(`Command ${i}: ${command}`);
console.log(`  PRNG calls: ${random_call_count - prev_count}`);
console.log(`  Total calls: ${random_call_count}`);
prev_count = random_call_count;
""")

if __name__ == "__main__":
    # Test isthmus with known seed
    debug_prng_tracking(
        width=1200,
        height=1000,
        cells_desired=10000,
        template="isthmus",
        seed="654321"
    )