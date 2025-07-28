#!/usr/bin/env python3
"""
Try to identify which template FMG used based on the height pattern.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.config.heightmap_templates import HEIGHTMAP_TEMPLATES
from py_fmg.utils.random import set_random_seed


def analyze_template_pattern(template_str):
    """Analyze what a template should produce."""
    lines = template_str.strip().split('\n')
    commands = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            commands.append(parts[0])
    return commands


def main():
    # Load FMG data
    with open('heights.json') as f:
        fmg_data = json.load(f)
    
    fmg_heights = np.array(fmg_data['heights']['data'])
    
    print("FMG Height Characteristics:")
    print(f"  Min: {fmg_heights.min()}")
    print(f"  Max: {fmg_heights.max()}")
    print(f"  Mean: {fmg_heights.mean():.1f}")
    print(f"  Land %: {np.sum(fmg_heights >= 20) / len(fmg_heights) * 100:.1f}%")
    print(f"  Has values 90-100: {np.any(fmg_heights >= 90)}")
    print(f"  Values >= 90: {np.sum(fmg_heights >= 90)}")
    
    # Check edge pattern
    edge_mean = np.mean(fmg_heights[:123])  # First row
    center_idx = len(fmg_heights) // 2
    center_mean = np.mean(fmg_heights[center_idx-500:center_idx+500])
    
    print(f"\nSpatial pattern:")
    print(f"  Edge mean: {edge_mean:.1f}")
    print(f"  Center mean: {center_mean:.1f}")
    print(f"  Center > Edge: {center_mean > edge_mean}")
    
    # Analyze all templates
    print("\n\nAnalyzing templates for clues:")
    print("=" * 60)
    
    for name, template_str in HEIGHTMAP_TEMPLATES.items():
        commands = analyze_template_pattern(template_str)
        
        # Count command types
        hills = commands.count('Hill')
        ranges = commands.count('Range')
        troughs = commands.count('Trough')
        pits = commands.count('Pit')
        straits = commands.count('Strait')
        has_multiply = 'Multiply' in commands
        has_mask = 'Mask' in commands
        
        # Check for multiply value
        multiply_val = None
        for line in template_str.strip().split('\n'):
            if line.strip().startswith('Multiply'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    multiply_val = float(parts[1])
        
        print(f"\n{name}:")
        print(f"  Hills: {hills}, Ranges: {ranges}, Troughs: {troughs}, Pits: {pits}, Straits: {straits}")
        print(f"  Has Multiply: {has_multiply} (value: {multiply_val})")
        print(f"  Has Mask: {has_mask}")
        
        # Templates that can produce high values (90-100)
        can_produce_high = (hills >= 5 or ranges >= 2) and (not has_multiply or multiply_val >= 0.8)
        print(f"  Can produce 90-100: {can_produce_high}")
    
    # Based on FMG having values up to 100 and 70% land, likely candidates are:
    print("\n\nMost likely templates based on FMG output:")
    print("- highIsland (has many hills, no multiply to reduce)")
    print("- continents (balanced land/water)")
    print("- isthmus (can produce high values)")
    
    # The seed might also give us a clue
    print(f"\n\nSeed analysis:")
    print(f"FMG seed: {fmg_data['seed']}")
    print("This looks like a random seed, suggesting FMG used 'random template'")


if __name__ == "__main__":
    main()