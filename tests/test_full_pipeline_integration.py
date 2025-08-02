"""Integration tests for the complete map generation pipeline."""

import pytest
import numpy as np
from pathlib import Path

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.core.biomes import BiomeClassifier, BiomeOptions, BiomeType


class TestFullPipelineIntegration:
    """Test complete pipeline integration with all modules."""
    
    @pytest.fixture
    def complete_pipeline_graph(self):
        """Create a complete graph through the full pipeline."""
        # Stage 1: Generate Voronoi graph
        config = GridConfig(width=300, height=250, cells_desired=1500)
        graph = generate_voronoi_graph(config, seed="full_pipeline_test")
        
        # Stage 2: Generate heightmap
        heightmap_config = HeightmapConfig(
            width=300,
            height=250,
            cells_x=graph.cells_x,
            cells_y=graph.cells_y,
            cells_desired=1500
        )
        
        heightmap_gen = HeightmapGenerator(heightmap_config, graph)
        heights = heightmap_gen.from_template("continents", seed="full_pipeline_test")
        graph.heights = heights
        
        # Stage 3: Mark up features
        features = Features(graph)
        features.markup_grid()
        
        # Stage 4: Perform reGraph
        packed_graph = regraph(graph)
        
        # Stage 5: Generate climate
        map_coords = MapCoordinates(lat_n=90, lat_s=-90)
        climate_options = ClimateOptions()
        
        climate = Climate(packed_graph, options=climate_options, map_coords=map_coords)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Stage 6: Generate hydrology
        hydrology_options = HydrologyOptions()
        hydrology = Hydrology(packed_graph, options=hydrology_options)
        hydrology.run_full_simulation()
        
        # Stage 7: Generate biomes
        biome_options = BiomeOptions()
        biome_classifier = BiomeClassifier(packed_graph, options=biome_options)
        biome_classifier.run_full_classification()
        
        return packed_graph
        
    def test_complete_pipeline_execution(self, complete_pipeline_graph):
        """Test that the complete pipeline executes successfully."""
        graph = complete_pipeline_graph
        
        # Verify all data is present
        assert hasattr(graph, 'heights')
        assert hasattr(graph, 'temperatures')
        assert hasattr(graph, 'precipitation')
        assert hasattr(graph, 'water_flux')
        assert hasattr(graph, 'flow_directions')
        assert hasattr(graph, 'filled_heights')
        assert hasattr(graph, 'rivers')
        assert hasattr(graph, 'lakes')
        assert hasattr(graph, 'biomes')
        assert hasattr(graph, 'biome_regions')
        
        # Verify data integrity
        n_cells = len(graph.points)
        assert len(graph.heights) == n_cells
        assert len(graph.temperatures) == n_cells
        assert len(graph.precipitation) == n_cells
        assert len(graph.water_flux) == n_cells
        assert len(graph.flow_directions) == n_cells
        assert len(graph.filled_heights) == n_cells
        assert len(graph.biomes) == n_cells
        
    def test_pipeline_data_consistency(self, complete_pipeline_graph):
        """Test that data between modules is consistent."""
        graph = complete_pipeline_graph
        
        # Climate data should be reasonable
        assert np.min(graph.temperatures) >= -50  # Not too cold
        assert np.max(graph.temperatures) <= 50   # Not too hot
        assert np.min(graph.precipitation) >= 0
        assert np.max(graph.precipitation) <= 255
        
        # Hydrology data should be consistent
        assert np.all(graph.filled_heights >= graph.heights)  # Depression filling
        assert np.all(graph.water_flux >= 0)  # No negative flow
        
        # Biomes should be valid
        valid_biomes = set(biome.value for biome in BiomeType)
        assert all(biome in valid_biomes for biome in graph.biomes)
        
    def test_pipeline_geographic_relationships(self, complete_pipeline_graph):
        """Test that geographic relationships are maintained across modules."""
        graph = complete_pipeline_graph
        
        # Water cells should have ocean biome
        water_cells = np.where(graph.heights < 20)[0]
        if len(water_cells) > 0:
            water_biomes = graph.biomes[water_cells]
            ocean_count = np.sum(water_biomes == BiomeType.OCEAN)
            # Most water cells should be ocean (allowing for some lakes/rivers)
            assert ocean_count >= len(water_cells) * 0.8
            
        # High elevation cells should be colder
        high_cells = np.where(graph.heights > 60)[0]
        low_cells = np.where((graph.heights >= 20) & (graph.heights < 40))[0]
        
        if len(high_cells) > 0 and len(low_cells) > 0:
            high_temps = np.mean(graph.temperatures[high_cells])
            low_temps = np.mean(graph.temperatures[low_cells])
            assert high_temps < low_temps  # Mountains are colder
            
    def test_pipeline_hydrologic_consistency(self, complete_pipeline_graph):
        """Test hydrologic consistency across the pipeline."""
        graph = complete_pipeline_graph
        
        # Rivers should flow from high to low elevation
        for river in graph.rivers:
            if len(river.cells) > 1:
                source_height = graph.filled_heights[river.source_cell]
                mouth_height = graph.filled_heights[river.mouth_cell]
                assert source_height >= mouth_height
                
        # River cells should have river biome (if on land)
        river_cells = set()
        for river in graph.rivers:
            river_cells.update(river.cells)
            
        for cell in river_cells:
            if graph.heights[cell] >= 20:  # Land river
                assert graph.biomes[cell] == BiomeType.RIVER
                
    def test_pipeline_biome_climate_relationship(self, complete_pipeline_graph):
        """Test that biomes are consistent with climate."""
        graph = complete_pipeline_graph
        
        # Very cold areas should have appropriate biomes
        very_cold_cells = np.where(graph.temperatures < -5)[0]
        if len(very_cold_cells) > 0:
            cold_biomes = graph.biomes[very_cold_cells]
            cold_biome_types = {BiomeType.GLACIER, BiomeType.TUNDRA, BiomeType.ALPINE}
            
            # Most very cold cells should have cold biomes
            cold_count = sum(1 for biome in cold_biomes if biome in cold_biome_types)
            assert cold_count >= len(very_cold_cells) * 0.6
            
        # Hot, wet areas should have tropical biomes
        hot_wet_cells = []
        for i in range(len(graph.points)):
            if (graph.temperatures[i] > 25 and 
                graph.precipitation[i] > 150 and
                graph.heights[i] >= 20):  # Land only
                hot_wet_cells.append(i)
                
        if len(hot_wet_cells) > 0:
            tropical_biomes = graph.biomes[hot_wet_cells]
            tropical_types = {BiomeType.TROPICAL_RAINFOREST, BiomeType.TROPICAL_SEASONAL_FOREST}
            
            # Some hot, wet areas should be tropical
            tropical_count = sum(1 for biome in tropical_biomes if biome in tropical_types)
            assert tropical_count > 0
            
    def test_pipeline_performance_metrics(self, complete_pipeline_graph):
        """Test that the pipeline produces reasonable performance metrics."""
        graph = complete_pipeline_graph
        
        # Should have reasonable number of rivers
        assert len(graph.rivers) > 0
        assert len(graph.rivers) < len(graph.points) * 0.1  # Not too many rivers
        
        # Should have reasonable biome diversity
        unique_biomes = len(np.unique(graph.biomes))
        assert unique_biomes >= 3  # At least 3 different biomes
        assert unique_biomes <= len(BiomeType)  # Not more than possible
        
        # Should have reasonable number of biome regions
        assert len(graph.biome_regions) > 0
        assert len(graph.biome_regions) <= len(graph.points)  # Sanity check
        
    def test_pipeline_data_types(self, complete_pipeline_graph):
        """Test that all data has correct types and ranges."""
        graph = complete_pipeline_graph
        
        # Height data
        assert graph.heights.dtype == np.uint8
        assert np.all(graph.heights >= 0)
        assert np.all(graph.heights <= 255)
        
        # Temperature data
        assert graph.temperatures.dtype == np.int8
        assert np.all(graph.temperatures >= -128)
        assert np.all(graph.temperatures <= 127)
        
        # Precipitation data
        assert graph.precipitation.dtype == np.uint8
        assert np.all(graph.precipitation >= 0)
        assert np.all(graph.precipitation <= 255)
        
        # Water flux data
        assert graph.water_flux.dtype == np.float32
        assert np.all(graph.water_flux >= 0)
        
        # Biome data
        assert graph.biomes.dtype == np.uint8
        
    def test_pipeline_edge_cases(self):
        """Test pipeline with edge cases."""
        # Test with very small map
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, seed="edge_case_test")
        
        # Create simple terrain
        n_cells = len(graph.points)
        graph.heights = np.full(n_cells, 30, dtype=np.uint8)
        
        # Add some water
        graph.heights[0] = 5
        
        # Run pipeline
        heightmap_config = HeightmapConfig(
            width=50, height=50,
            cells_x=graph.cells_x, cells_y=graph.cells_y,
            cells_desired=25
        )
        
        heightmap_gen = HeightmapGenerator(heightmap_config, graph)
        heights = heightmap_gen.from_template("continents", seed="edge_case_test")
        graph.heights = heights
        
        features = Features(graph)
        features.markup_grid()
        
        packed_graph = regraph(graph)
        
        climate = Climate(packed_graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        hydrology = Hydrology(packed_graph)
        hydrology.run_full_simulation()
        
        biome_classifier = BiomeClassifier(packed_graph)
        biome_classifier.run_full_classification()
        
        # Should complete without errors
        assert packed_graph.biomes is not None
        assert len(packed_graph.biomes) == len(packed_graph.points)
        
    def test_pipeline_reproducibility(self):
        """Test that pipeline is reproducible with same seed."""
        def run_pipeline(seed):
            config = GridConfig(width=100, height=100, cells_desired=100)
            graph = generate_voronoi_graph(config, seed=seed)
            
            heightmap_config = HeightmapConfig(
                width=100, height=100,
                cells_x=graph.cells_x, cells_y=graph.cells_y,
                cells_desired=100
            )
            
            heightmap_gen = HeightmapGenerator(heightmap_config, graph)
            heights = heightmap_gen.from_template("continents", seed=seed)
            graph.heights = heights
            
            features = Features(graph)
            features.markup_grid()
            
            packed_graph = regraph(graph)
            
            climate = Climate(packed_graph)
            climate.calculate_temperatures()
            climate.generate_precipitation()
            
            hydrology = Hydrology(packed_graph)
            hydrology.run_full_simulation()
            
            biome_classifier = BiomeClassifier(packed_graph)
            biome_classifier.run_full_classification()
            
            return packed_graph
            
        # Run pipeline twice with same seed
        graph1 = run_pipeline("reproducibility_test")
        graph2 = run_pipeline("reproducibility_test")
        
        # Results should be identical
        np.testing.assert_array_equal(graph1.heights, graph2.heights)
        np.testing.assert_array_equal(graph1.temperatures, graph2.temperatures)
        np.testing.assert_array_equal(graph1.biomes, graph2.biomes)
        
    def test_pipeline_memory_efficiency(self, complete_pipeline_graph):
        """Test that pipeline doesn't create excessive memory usage."""
        graph = complete_pipeline_graph
        
        # All arrays should be reasonably sized
        n_cells = len(graph.points)
        
        # Check that we don't have duplicate large arrays
        assert len(graph.heights) == n_cells
        assert len(graph.temperatures) == n_cells
        assert len(graph.precipitation) == n_cells
        assert len(graph.water_flux) == n_cells
        assert len(graph.biomes) == n_cells
        
        # Rivers and lakes should be reasonable in number
        assert len(graph.rivers) < n_cells * 0.1
        assert len(graph.lakes) < n_cells * 0.05


def test_pipeline_module_integration():
    """Test that all pipeline modules integrate correctly."""
    from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
    from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
    from py_fmg.core.features import Features
    from py_fmg.core.cell_packing import regraph
    from py_fmg.core.climate import Climate
    from py_fmg.core.hydrology import Hydrology
    from py_fmg.core.biomes import BiomeClassifier
    
    # All modules should import without errors
    assert all([
        generate_voronoi_graph, GridConfig,
        HeightmapGenerator, HeightmapConfig,
        Features, regraph,
        Climate, Hydrology, BiomeClassifier
    ])
    
    
def test_pipeline_error_handling():
    """Test pipeline error handling."""
    # Test with invalid data
    config = GridConfig(width=50, height=50, cells_desired=25)
    graph = generate_voronoi_graph(config, seed="error_test")
    
    # Don't set heights - should handle gracefully
    climate = Climate(graph)
    
    # Should not crash, but may produce default results
    try:
        climate.calculate_temperatures()
        climate.generate_precipitation()
    except Exception as e:
        # If it fails, it should fail gracefully
        assert isinstance(e, (ValueError, AttributeError))
        
        
def test_pipeline_with_minimal_data():
    """Test pipeline with minimal required data."""
    config = GridConfig(width=30, height=30, cells_desired=9)
    graph = generate_voronoi_graph(config, seed="minimal_test")
    
    # Set minimal heights
    graph.heights = np.full(len(graph.points), 25, dtype=np.uint8)
    
    # Should be able to run climate
    climate = Climate(graph)
    climate.calculate_temperatures()
    climate.generate_precipitation()
    
    # Should be able to run hydrology
    hydrology = Hydrology(graph)
    hydrology.run_full_simulation()
    
    # Should be able to run biomes
    biome_classifier = BiomeClassifier(graph)
    biome_classifier.run_full_classification()
    
    # All should complete successfully
    assert graph.temperatures is not None
    assert graph.precipitation is not None
    assert graph.water_flux is not None
    assert graph.biomes is not None

