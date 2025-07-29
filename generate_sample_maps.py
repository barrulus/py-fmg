#!/usr/bin/env python3
"""
Generate high-resolution heightmaps with custom seed.

Usage:
    python generate_sample_maps.py [seed]

If no seed is provided, defaults to "default_seed"
"""

import sys
from visualize_heightmap import create_simple_height_image

# Get seed from command line argument or use default
seed = sys.argv[1] if len(sys.argv) > 1 else "default_seed"

# Generate samples of the most interesting templates
templates = [
    "volcano",
    "lowIsland",
    "continents",
    "pangea",
    "atoll",
    "archipelago",
    "mediterranean",
    "peninsula",
    "isthmus",
    "fractious",
]

print(f"Generating high-resolution maps with seed: {seed}")

for template in templates:
    print(f"\nGenerating {template} template...")
    create_simple_height_image(
        width=1200,
        height=1000,
        cells_desired=10000,
        template=template,
        seed=seed,
    )

print("\nAll maps generated!")
