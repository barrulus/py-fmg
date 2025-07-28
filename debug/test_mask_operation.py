#!/usr/bin/env python3
"""
Test the mask operation to see if it's causing excessive height reduction.
"""

import numpy as np

def test_mask_formula():
    """Test the mask formula with different values."""
    
    print("🔍 MASK OPERATION TEST")
    print("=" * 50)
    
    # Test height value
    h = 52.0  # Max height before mask
    power = 4  # Mask power from template
    factor = abs(power)
    
    print(f"\nTesting with height={h}, power={power}")
    print(f"Formula: new_h = (h * (factor - 1) + masked) / factor")
    print(f"         where masked = h * distance")
    print(f"         and distance = (1 - nx²) * (1 - ny²)")
    print()
    
    # Test different positions
    positions = [
        (0.5, 0.5, "center"),
        (0.7, 0.5, "mid-right"),
        (0.9, 0.5, "near-edge"),
        (0.95, 0.5, "very-near-edge"),
        (1.0, 0.5, "edge")
    ]
    
    print("Position tests:")
    for x_norm, y_norm, desc in positions:
        # Convert to [-1, 1] range
        nx = 2 * x_norm - 1
        ny = 2 * y_norm - 1
        
        # Calculate distance
        distance = (1 - nx**2) * (1 - ny**2)
        
        # Apply mask
        masked = h * distance
        new_h = (h * (factor - 1) + masked) / factor
        
        print(f"  {desc:15} x={x_norm:.2f}, y={y_norm:.2f}")
        print(f"    nx={nx:+.2f}, ny={ny:+.2f}")
        print(f"    distance = {distance:.4f}")
        print(f"    masked = {h} * {distance:.4f} = {masked:.2f}")
        print(f"    new_h = ({h} * 3 + {masked:.2f}) / 4 = {new_h:.2f}")
        print(f"    reduction: {h - new_h:.2f} ({(h - new_h)/h*100:.1f}%)")
        print()
    
    # Find position that would reduce 52 to ~45
    target = 45
    print(f"\nReverse calculation: what distance gives {target} from {h}?")
    # new_h = (h * (factor - 1) + h * distance) / factor
    # new_h = h * (factor - 1 + distance) / factor
    # new_h * factor = h * (factor - 1 + distance)
    # distance = (new_h * factor / h) - factor + 1
    distance_needed = (target * factor / h) - factor + 1
    print(f"  Distance needed: {distance_needed:.4f}")
    
    # What position gives this distance?
    # distance = (1 - nx²) * (1 - ny²)
    # If ny = 0 (center in y), then distance = 1 - nx²
    # nx² = 1 - distance
    # nx = sqrt(1 - distance)
    if distance_needed > 0:
        nx_needed = np.sqrt(1 - distance_needed)
        x_norm_needed = (nx_needed + 1) / 2
        print(f"  At y=0.5 (center), this requires nx={nx_needed:.3f}, or x_norm={x_norm_needed:.3f}")
        print(f"  This is {x_norm_needed * 100:.1f}% from left edge")

if __name__ == "__main__":
    test_mask_formula()