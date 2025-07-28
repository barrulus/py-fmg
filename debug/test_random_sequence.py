#!/usr/bin/env python3
"""
Test random number generation to ensure it matches FMG's sequence.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_fmg.core.alea_prng import AleaPRNG

def test_random_sequence():
    """Compare random number sequences."""
    
    print("🔍 RANDOM NUMBER SEQUENCE TEST")
    print("=" * 50)
    
    # Test with the actual seed used
    seed = "651658815"
    prng = AleaPRNG(seed)
    
    print(f"Testing with seed: {seed}")
    print("\nFirst 10 random numbers:")
    
    # Generate some random numbers
    for i in range(10):
        val = prng.random()
        print(f"  {i+1}: {val:.10f}")
    
    # Test specific operations used in heightmap generation
    print("\nTesting specific random operations:")
    
    # Reset seed for consistent test
    prng = AleaPRNG(seed)
    
    # Test range parsing (used in getNumberInRange)
    print("\n1. Range parsing (90-99):")
    for i in range(5):
        val = 90 + prng.random() * 9
        print(f"   {val:.3f}")
    
    # Test blob spreading random factor
    print("\n2. Blob spreading factor (random * 0.2 + 0.9):")
    for i in range(5):
        val = prng.random() * 0.2 + 0.9
        print(f"   {val:.3f}")
    
    # Test smooth operation random
    print("\n3. Smooth operation (random * 0.2 + 0.9):")
    for i in range(5):
        val = prng.random() * 0.2 + 0.9
        print(f"   {val:.3f}")

if __name__ == "__main__":
    test_random_sequence()