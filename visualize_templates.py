#!/usr/bin/env python3
"""
Quick script to visualize different heightmap templates.
"""

from visualize_heightmap import create_simple_height_image

# Generate visualizations for different templates
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

for template in templates:
    print(f"\nGenerating high-res {template} template...")
    create_simple_height_image(
        width=800,
        height=800,
        cells_desired=10000,
        template=template,
        seed=f"{template}_demo",
    )

print("\nAll visualizations complete!")
