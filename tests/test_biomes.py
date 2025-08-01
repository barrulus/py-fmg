"""Tests for biomes classification module."""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.biomes import BiomeClassifier, BiomeOptions, BiomeType, BIOME_NAMES
from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates
from py_fmg.core.hydrology import Hydrology, HydrologyOptions


class TestBiomes:
    """Test biome classification."""
    
    @pytest.fixture
    def climate_graph(self):
        """Create a test graph with climate data."""
        config = GridConfig(200, 200, 400)
        graph = generate_voronoi_graph(config, seed="biome_test")
        
        n_cells = len(graph.points)
        
        # Create varied terrain
        graph.heights = np.full(n_cells, 40, dtype=np.uint8)
        
        # Add water, land, and mountains
        for i in range(n_cells):
            x, y = graph.points[i]
            
            # Ocean at edges
            if x < 20 or x > 180 or y < 20 or y > 180:
                graph.heights[i] = 5
            # Mountains in center
            elif 80 < x < 120 and 80 < y < 120:
                graph.heights[i] = 80
                
        # Generate climate data
        climate = Climate(graph, map_coords=MapCoordinates(lat_n=60, lat_s=-60))
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        return graph
        
    @pytest.fixture
    def full_pipeline_graph(self):
        """Create a graph with full pipeline data (climate + hydrology)."""
        config = GridConfig(150, 150, 300)
        graph = generate_voronoi_graph(config, seed="biome_full_test")
        
        n_cells = len(graph.points)
        graph.heights = np.full(n_cells, 35, dtype=np.uint8)
        
        # Create diverse terrain
        for i in range(n_cells):
            x, y = graph.points[i]
            
            # Ocean
            if y > 130:
                graph.heights[i] = 5
            # River valley
            elif 70 < x < 80:
                graph.heights[i] = 25
            # Mountains
            elif x > 120:
                graph.heights[i] = 75
                
        # Generate climate
        climate = Climate(graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Generate hydrology
        hydrology = Hydrology(graph)
        hydrology.run_full_simulation()
        
        return graph
        
    def test_biome_classifier_initialization(self, climate_graph):
        """Test biome classifier initialization."""
        classifier = BiomeClassifier(climate_graph)
        
        assert classifier.graph is climate_graph
        assert classifier.options is not None
        assert classifier.biomes is None
        assert len(classifier.biome_regions) == 0
        
        # Check biome matrix is initialized
        assert classifier.biome_matrix is not None
        assert len(classifier.biome_matrix) == 6  # 6 temperature bands
        assert all(len(row) == 5 for row in classifier.biome_matrix)  # 5 precipitation bands
        
    def test_temperature_band_classification(self, climate_graph):
        """Test temperature band classification."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test temperature band mapping
        assert classifier._get_temperature_band(-15) == 0  # Very cold
        assert classifier._get_temperature_band(0) == 1   # Cold
        assert classifier._get_temperature_band(10) == 2  # Cool
        assert classifier._get_temperature_band(20) == 3  # Temperate
        assert classifier._get_temperature_band(30) == 4  # Warm
        assert classifier._get_temperature_band(40) == 5  # Hot
        
    def test_precipitation_band_classification(self, climate_graph):
        """Test precipitation band classification."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test precipitation band mapping
        assert classifier._get_precipitation_band(10) == 0   # Very dry
        assert classifier._get_precipitation_band(30) == 1   # Dry
        assert classifier._get_precipitation_band(75) == 2   # Moderate
        assert classifier._get_precipitation_band(150) == 3  # Wet
        assert classifier._get_precipitation_band(200) == 4  # Very wet
        
    def test_basic_biome_classification(self, climate_graph):
        """Test basic biome classification."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Should have biomes assigned
        assert classifier.biomes is not None
        assert len(classifier.biomes) == len(climate_graph.points)
        
        # Should have ocean biomes for water cells
        water_cells = np.where(climate_graph.heights < 20)[0]
        if len(water_cells) > 0:
            water_biomes = classifier.biomes[water_cells]
            assert np.all(water_biomes == BiomeType.OCEAN)
            
        # Should have land biomes for land cells
        land_cells = np.where(climate_graph.heights >= 20)[0]
        if len(land_cells) > 0:
            land_biomes = classifier.biomes[land_cells]
            assert np.all(land_biomes != BiomeType.OCEAN)
            
    def test_alpine_biome_classification(self, climate_graph):
        """Test alpine biome classification for high elevations."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # High elevation cells should be alpine
        high_cells = np.where(climate_graph.heights >= classifier.options.alpine_height_threshold)[0]
        
        if len(high_cells) > 0:
            alpine_biomes = classifier.biomes[high_cells]
            # Most high elevation cells should be alpine
            alpine_count = np.sum(alpine_biomes == BiomeType.ALPINE)
            assert alpine_count > len(high_cells) * 0.5
            
    def test_glacier_biome_classification(self, climate_graph):
        """Test glacier biome classification for very cold areas."""
        # Manually set some cells to be very cold
        very_cold_cells = []
        for i in range(min(10, len(climate_graph.temperatures))):
            climate_graph.temperatures[i] = -15
            very_cold_cells.append(i)
            
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Very cold cells should be glaciers (if not water)
        for cell in very_cold_cells:
            if climate_graph.heights[cell] >= 20:  # Land cell
                assert classifier.biomes[cell] == BiomeType.GLACIER
                
    def test_biome_matrix_lookup(self, climate_graph):
        """Test biome matrix lookup functionality."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test specific temperature/precipitation combinations
        # Temperate (band 3) + moderate precipitation (band 2) = temperate deciduous forest
        biome = classifier.biome_matrix[3][2]
        assert biome == BiomeType.TEMPERATE_DECIDUOUS_FOREST
        
        # Hot (band 5) + very wet (band 4) = tropical rainforest
        biome = classifier.biome_matrix[5][4]
        assert biome == BiomeType.TROPICAL_RAINFOREST
        
        # Cold (band 1) + dry (band 1) = tundra
        biome = classifier.biome_matrix[1][1]
        assert biome == BiomeType.TUNDRA
        
    def test_coastal_influences(self, climate_graph):
        """Test coastal influences on biomes."""
        # Add distance field for coastal effects
        n_cells = len(climate_graph.points)
        climate_graph.distance_field = np.zeros(n_cells)
        
        # Set some cells as coastal
        coastal_cells = []
        for i in range(n_cells):
            if climate_graph.heights[i] >= 20:  # Land cell
                # Check if near water
                for neighbor in climate_graph.cell_neighbors[i]:
                    if climate_graph.heights[neighbor] < 20:
                        climate_graph.distance_field[i] = 1.0  # Close to coast
                        coastal_cells.append(i)
                        break
                        
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Should complete without errors
        assert classifier.biomes is not None
        
    def test_river_influences(self, full_pipeline_graph):
        """Test river influences on biomes."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.classify_biomes()
        
        # River cells should be marked as rivers
        if hasattr(full_pipeline_graph, 'rivers') and full_pipeline_graph.rivers:
            for river in full_pipeline_graph.rivers:
                for cell in river.cells:
                    if full_pipeline_graph.heights[cell] >= 20:  # Land river
                        assert classifier.biomes[cell] == BiomeType.RIVER
                        
    def test_wetland_formation(self, full_pipeline_graph):
        """Test wetland formation in high water flux areas."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.classify_biomes()
        
        # Should complete without errors
        assert classifier.biomes is not None
        
        # Check if wetlands formed where appropriate
        wetland_cells = np.where(classifier.biomes == BiomeType.WETLAND)[0]
        
        # If wetlands formed, they should be in high water flux areas
        if len(wetland_cells) > 0 and hasattr(full_pipeline_graph, 'water_flux'):
            wetland_flux = [full_pipeline_graph.water_flux[cell] for cell in wetland_cells]
            avg_wetland_flux = np.mean(wetland_flux)
            avg_total_flux = np.mean(full_pipeline_graph.water_flux)
            
            # Wetlands should have higher than average water flux
            assert avg_wetland_flux >= avg_total_flux
            
    def test_biome_region_generation(self, climate_graph):
        """Test biome region generation."""
        classifier = BiomeClassifier(climate_graph)
        classifier.generate_biome_regions()
        
        # Should have generated regions
        assert len(classifier.biome_regions) > 0
        
        # Check region properties
        total_cells = set()
        for region in classifier.biome_regions:
            assert region.id >= 0
            assert isinstance(region.biome_type, BiomeType)
            assert len(region.cells) > 0
            assert region.area > 0
            assert region.center_cell in region.cells
            
            # Regions shouldn't overlap
            assert len(total_cells.intersection(region.cells)) == 0
            total_cells.update(region.cells)
            
        # All cells should be in some region
        assert len(total_cells) == len(climate_graph.points)
        
    def test_connected_biome_cells(self, climate_graph):
        """Test finding connected biome cells."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        # Test with a specific cell
        test_cell = 0
        biome_type = BiomeType(classifier.biomes[test_cell])
        processed = set()
        
        connected_cells = classifier._find_connected_biome_cells(test_cell, biome_type, processed)
        
        # Should find at least the test cell itself
        assert test_cell in connected_cells
        
        # All connected cells should have the same biome type
        for cell in connected_cells:
            assert BiomeType(classifier.biomes[cell]) == biome_type
            
    def test_region_center_calculation(self, climate_graph):
        """Test region center calculation."""
        classifier = BiomeClassifier(climate_graph)
        
        # Test with a small set of cells
        test_cells = {0, 1, 2}
        center_cell = classifier._find_region_center(test_cells)
        
        # Center should be one of the cells in the region
        assert center_cell in test_cells
        
    def test_full_classification_pipeline(self, full_pipeline_graph):
        """Test the complete biome classification pipeline."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.run_full_classification()
        
        # All components should be generated
        assert classifier.biomes is not None
        assert len(classifier.biome_regions) > 0
        
        # Data should be stored on graph
        assert hasattr(full_pipeline_graph, 'biomes')
        assert hasattr(full_pipeline_graph, 'biome_regions')
        
    def test_biome_statistics(self, climate_graph):
        """Test biome statistics generation."""
        classifier = BiomeClassifier(climate_graph)
        classifier.classify_biomes()
        
        stats = classifier.get_biome_statistics()
        
        # Should have statistics
        assert len(stats) > 0
        
        # All values should be positive
        for biome_name, count in stats.items():
            assert biome_name in BIOME_NAMES.values()
            assert count > 0
            
        # Total count should match number of cells
        total_count = sum(stats.values())
        assert total_count == len(climate_graph.points)
        
    def test_custom_biome_options(self, climate_graph):
        """Test biome classification with custom options."""
        custom_options = BiomeOptions(
            coastal_effect_distance=5.0,
            river_effect_distance=3.0,
            alpine_height_threshold=60
        )
        
        classifier = BiomeClassifier(climate_graph, options=custom_options)
        classifier.classify_biomes()
        
        # Should use custom options
        assert classifier.options.coastal_effect_distance == 5.0
        assert classifier.options.river_effect_distance == 3.0
        assert classifier.options.alpine_height_threshold == 60
        
        # Should complete successfully
        assert classifier.biomes is not None
        
    def test_biome_diversity(self, full_pipeline_graph):
        """Test that classification produces diverse biomes."""
        classifier = BiomeClassifier(full_pipeline_graph)
        classifier.run_full_classification()
        
        # Should have multiple biome types
        unique_biomes = np.unique(classifier.biomes)
        assert len(unique_biomes) >= 3  # At least 3 different biomes
        
        # Should include both water and land biomes
        has_water = BiomeType.OCEAN in unique_biomes
        has_land = any(biome != BiomeType.OCEAN for biome in unique_biomes)
        
        assert has_water and has_land
        
    def test_biome_names_completeness(self):
        """Test that all biome types have names."""
        for biome_type in BiomeType:
            assert biome_type in BIOME_NAMES
            assert isinstance(BIOME_NAMES[biome_type], str)
            assert len(BIOME_NAMES[biome_type]) > 0


def test_biome_module_imports():
    """Test that biome module imports correctly."""
    from py_fmg.core.biomes import BiomeClassifier, BiomeOptions, BiomeType, BIOME_NAMES
    
    assert BiomeClassifier is not None
    assert BiomeOptions is not None
    assert BiomeType is not None
    assert BIOME_NAMES is not None
    
    
def test_biome_type_enum():
    """Test BiomeType enum values."""
    # Test that enum values are as expected
    assert BiomeType.OCEAN == 0
    assert BiomeType.LAKE == 1
    assert BiomeType.TROPICAL_RAINFOREST == 15
    
    # Test that all values are unique
    values = [biome.value for biome in BiomeType]
    assert len(values) == len(set(values))
    
    
def test_biome_options_defaults():
    """Test BiomeOptions default values."""
    options = BiomeOptions()
    
    assert options.coastal_effect_distance == 3.0
    assert options.river_effect_distance == 2.0
    assert options.wetland_threshold == 0.8
    assert options.alpine_height_threshold == 70
    assert options.glacier_temperature_threshold == -10

