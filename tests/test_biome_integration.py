"""
Test biome integration with the FMG pipeline

Tests that biome classification works correctly with real climate and hydrology data.
"""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.climate import Climate
from py_fmg.core.hydrology import Hydrology
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cell_packing import regraph


def test_biome_integration_with_pipeline():
    """Test that biomes integrate correctly with the complete pipeline"""
    
    # Generate a small test world
    config = GridConfig(width=400, height=300, cells_desired=1000)
    voronoi_graph = generate_voronoi_graph(config, seed="biome-test")
    
    # Generate heightmap
    heightmap_config = HeightmapConfig(
        width=400, height=300,
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=1000,
        spacing=voronoi_graph.spacing,
    )
    
    heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph, seed="biome-test")
    heights = heightmap_gen.from_template("atoll", seed="biome-test")
    voronoi_graph.heights = heights
    
    # Generate features and pack
    features = Features(voronoi_graph, seed="biome-test")
    features.markup_grid()
    packed_graph = regraph(voronoi_graph)
    
    # Generate climate
    climate = Climate(packed_graph)
    climate.calculate_temperatures()
    climate.generate_precipitation()
    
    # Generate hydrology
    packed_features = Features(packed_graph, seed="biome-test")
    packed_features.markup_grid()
    hydrology = Hydrology(packed_graph, packed_features, climate)
    rivers = hydrology.generate_rivers()
    
    # Test biome classification
    biome_classifier = BiomeClassifier()
    
    # Prepare river data
    has_river = np.zeros(len(packed_graph.points), dtype=bool)
    river_flux = np.zeros(len(packed_graph.points), dtype=float)
    
    for river_id, river_data in rivers.items():
        for cell_id in river_data.cells:
            if cell_id < len(has_river):
                has_river[cell_id] = True
                river_flux[cell_id] = max(river_flux[cell_id], river_data.discharge)
    
    # Classify biomes
    biomes = biome_classifier.classify_biomes(
        temperatures=climate.temperatures,
        precipitation=climate.precipitation,
        heights=packed_graph.heights,
        river_flux=river_flux,
        has_river=has_river
    )
    
    # Verify results
    assert len(biomes) == len(packed_graph.points)
    assert biomes.dtype == np.uint8
    assert np.all(biomes >= 0)
    assert np.all(biomes <= 12)
    
    # Should have water (marine) biomes
    assert np.any(biomes == 0)  # Marine
    
    # Should have some land biomes for atoll template
    land_biomes = biomes[packed_graph.heights >= 20]
    if len(land_biomes) > 0:
        assert np.any(land_biomes > 0)  # Some non-marine biomes
    
    # Verify biome properties
    unique_biomes = np.unique(biomes)
    for biome_id in unique_biomes:
        name = biome_classifier.get_biome_name(biome_id)
        color = biome_classifier.get_biome_color(biome_id)
        props = biome_classifier.get_biome_properties(biome_id)
        
        assert name != "Unknown"
        assert color.startswith("#")
        assert len(color) == 7  # Valid hex color
        assert "name" in props
        assert "habitability" in props
        assert "movement_cost" in props
    
    # Test river influence on biomes
    if np.any(has_river):
        river_cells = biomes[has_river]
        # River cells should have appropriate biomes (not necessarily different, 
        # but should be valid classifications)
        assert np.all(river_cells >= 0)
        assert np.all(river_cells <= 12)
    
    print(f"✓ Biome integration test passed with {len(unique_biomes)} biome types")
    print(f"  - Total cells: {len(biomes)}")
    print(f"  - Marine cells: {np.sum(biomes == 0)}")
    print(f"  - Land biomes: {len(unique_biomes) - (1 if 0 in unique_biomes else 0)}")
    
    return {
        "biomes": biomes,
        "biome_classifier": biome_classifier,
        "climate": climate,
        "rivers": rivers,
        "packed_graph": packed_graph
    }


def test_biome_with_different_climates():
    """Test biome classification with various climate conditions"""
    
    biome_classifier = BiomeClassifier()
    
    # Create test data with different climate scenarios
    n_cells = 100
    
    # Test scenario 1: Hot and dry (should get hot desert)
    hot_dry_biomes = biome_classifier.classify_biomes(
        temperatures=np.full(n_cells, 30.0),  # Hot
        precipitation=np.full(n_cells, 2.0),  # Very dry
        heights=np.full(n_cells, 50.0),       # Land
        has_river=np.zeros(n_cells, dtype=bool)  # No rivers
    )
    
    # Should be mostly hot desert
    assert np.sum(hot_dry_biomes == 1) > n_cells * 0.8  # At least 80% hot desert
    
    # Test scenario 2: Cold and wet (should get different biomes)
    cold_wet_biomes = biome_classifier.classify_biomes(
        temperatures=np.full(n_cells, -10.0),  # Cold
        precipitation=np.full(n_cells, 30.0),  # Wet
        heights=np.full(n_cells, 50.0),        # Land
    )
    
    # Should be mostly tundra or permafrost
    cold_biomes = np.sum((cold_wet_biomes == 10) | (cold_wet_biomes == 11))  # Tundra or Glacier
    assert cold_biomes > n_cells * 0.5  # At least 50% cold biomes
    
    # Test scenario 3: Tropical rainforest conditions (avoid wetland trigger)
    tropical_biomes = biome_classifier.classify_biomes(
        temperatures=np.full(n_cells, 25.0),   # Hot
        precipitation=np.full(n_cells, 20.0),  # Wet but not wetland-triggering
        heights=np.full(n_cells, 80.0),        # Higher land to avoid wetland conditions
    )
    
    # Should have tropical biomes (moisture = 24, temp = 25 -> should be tropical rainforest)
    tropical_count = np.sum((tropical_biomes == 7) | (tropical_biomes == 5))  # Tropical rainforest or seasonal
    assert tropical_count > 0  # Should have some tropical biomes
    
    print("✓ Different climate scenarios produce expected biome distributions")


if __name__ == "__main__":
    test_biome_integration_with_pipeline()
    test_biome_with_different_climates()
    print("All biome integration tests passed!")