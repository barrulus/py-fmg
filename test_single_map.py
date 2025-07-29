#!/usr/bin/env python3
"""Test script to generate a single map with the new features."""

from visualize_heightmap import create_simple_height_image

# Generate a single map to test the new features
create_simple_height_image(
    width=800,
    height=800,
    cells_desired=5000,
    template="archipelago",
    seed="test_seed_123"
)

print("Test map generated!")