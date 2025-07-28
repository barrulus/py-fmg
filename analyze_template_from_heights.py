#!/usr/bin/env python3
"""
Analyze FMG heights to determine which template was used.
"""

import json
import numpy as np
from py_fmg.config.heightmap_templates import TEMPLATES as HEIGHTMAP_TEMPLATES


def analyze_heights(heights):
    """Analyze height characteristics."""
    heights_array = np.array(heights)
    
    # Key characteristics
    min_h = heights_array.min()
    max_h = heights_array.max()
    mean_h = heights_array.mean()
    land_pct = np.sum(heights_array >= 20) / len(heights_array) * 100
    
    # Check for multiply effect (would cap max height)
    has_low_max = max_h < 50  # Multiply 0.4 would cap at ~40
    
    # Distribution characteristics
    very_high = np.sum(heights_array >= 80)
    high = np.sum(heights_array >= 50)
    
    return {
        'min': min_h,
        'max': max_h,
        'mean': mean_h,
        'land_pct': land_pct,
        'very_high_cells': very_high,
        'high_cells': high,
        'has_low_max': has_low_max
    }


def score_template(template_str, target_chars):
    """Score how well a template matches target characteristics."""
    score = 0
    
    # Parse template commands
    lines = template_str.strip().split('\n')
    has_multiply = False
    multiply_val = 1.0
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        if parts[0] == 'Multiply':
            has_multiply = True
            multiply_val = float(parts[1])
    
    # Check if template can produce high values
    if target_chars['max'] > 90:
        # Need no multiply or high multiply value
        if not has_multiply or multiply_val >= 0.8:
            score += 10
        else:
            score -= 20
    
    # Check land percentage
    # Templates with many hills tend to have 60-80% land
    # Templates with pits/troughs have less land
    land_diff = abs(target_chars['land_pct'] - 70)
    score -= land_diff / 10
    
    return score


def main():
    # Load FMG data
    with open('heights.json') as f:
        fmg_data = json.load(f)
    
    heights = fmg_data['heights']['data']
    
    # Analyze FMG heights
    fmg_chars = analyze_heights(heights)
    
    print("FMG Height Characteristics:")
    print(f"  Min: {fmg_chars['min']}")
    print(f"  Max: {fmg_chars['max']}")
    print(f"  Mean: {fmg_chars['mean']:.1f}")
    print(f"  Land %: {fmg_chars['land_pct']:.1f}%")
    print(f"  Cells >= 80: {fmg_chars['very_high_cells']}")
    print(f"  Cells >= 50: {fmg_chars['high_cells']}")
    
    # Score each template
    print("\n\nTemplate Scores (higher = better match):")
    print("=" * 60)
    
    scores = []
    for name, template_str in HEIGHTMAP_TEMPLATES.items():
        score = score_template(template_str, fmg_chars)
        scores.append((score, name))
    
    # Sort by score
    scores.sort(reverse=True)
    
    for score, name in scores[:5]:
        print(f"{name:20} Score: {score:6.1f}")
        
        # Show key template features
        template_str = HEIGHTMAP_TEMPLATES[name]
        lines = template_str.strip().split('\n')
        
        hills = sum(1 for line in lines if line.startswith('Hill'))
        pits = sum(1 for line in lines if line.startswith('Pit'))
        has_multiply = any(line.startswith('Multiply') for line in lines)
        
        if has_multiply:
            for line in lines:
                if line.startswith('Multiply'):
                    multiply_val = float(line.split()[1])
                    print(f"  - Has Multiply {multiply_val}")
        
        print(f"  - {hills} Hills, {pits} Pits")
    
    print("\n\nMost likely template:", scores[0][1])
    
    # Additional analysis - check if it could be a non-template heightmap
    print("\n\nChecking non-template options:")
    print("The heightmap type was 'fractious' - this might be a special mode")
    print("that generates heights differently than templates.")


if __name__ == "__main__":
    main()