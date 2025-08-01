#!/usr/bin/env python3
"""
Test to compare our Alea PRNG implementation with FMG's expected values.
"""

from py_fmg.core.alea_prng import AleaPRNG

def test_prng_output():
    """Test PRNG output for the specific seed used in FMG."""
    seed = "469833767"
    prng = AleaPRNG(seed)
    
    print(f"Testing Alea PRNG with seed: {seed}")
    print("First 10 random values:")
    
    for i in range(10):
        value = prng.random()
        print(f"{i+1}: {value:.10f}")
    
    # Test jittered grid generation
    print("\nTesting jittered grid generation:")
    prng = AleaPRNG(seed)  # Reset
    
    spacing = 10.95
    radius = spacing / 2
    jittering = radius * 0.9
    double_jittering = jittering * 2
    
    print(f"Spacing: {spacing}")
    print(f"Radius: {radius}")
    print(f"Jittering: {jittering}")
    print(f"Double jittering: {double_jittering}")
    
    # Generate first few jittered points
    print("\nFirst few jittered points:")
    y = radius
    x = radius
    
    for i in range(5):
        # Jitter calculation matching FMG
        xj = prng.random() * double_jittering - jittering
        yj = prng.random() * double_jittering - jittering
        
        final_x = round(x + xj, 2)
        final_y = round(y + yj, 2)
        
        print(f"Point {i+1}: ({final_x}, {final_y})")
        print(f"  Random values: {xj:.10f}, {yj:.10f}")
        
        x += spacing
        if x >= 1200:  # Width boundary
            x = radius
            y += spacing

if __name__ == "__main__":
    test_prng_output()