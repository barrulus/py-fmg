"""Integration tests for climate system in the full pipeline."""

import pytest
import numpy as np
from pathlib import Path

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates


class TestClimateIntegration:
    """Test climate integration in the full map generation pipeline."""
    
    @pytest.fixture
    def full_pipeline_graph(self):
        """Create a complete graph through the full pipeline up to climate."""
        # Stage 1: Generate Voronoi graph
        config = GridConfig(width=400, height=300, cells_desired=2000)
        graph = generate_voronoi_graph(config, seed="climate_integration_test")
        
        # Stage 2: Generate heightmap
        heightmap_config = HeightmapConfig(
            width=400,
            height=300,
            cells_x=graph.cells_x,
            cells_y=graph.cells_y,
            cells_desired=2000
        )
        
        heightmap_gen = HeightmapGenerator(heightmap_config, graph)
        heights = heightmap_gen.from_template("continents", seed="climate_integration_test")
        graph.heights = heights
        
        # Stage 3: Mark up features
        features = Features(graph)
        features.markup_grid()
        
        # Stage 4: Perform reGraph
        packed_graph = regraph(graph)
        
        return packed_graph
        
    def test_climate_integration_basic(self, full_pipeline_graph):
        """Test basic climate integration with full pipeline."""
        # Stage 5: Generate climate
        map_coords = MapCoordinates(lat_n=90, lat_s=-90)
        climate_options = ClimateOptions()
        
        climate = Climate(full_pipeline_graph, options=climate_options, map_coords=map_coords)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Verify climate data was added to graph
        assert hasattr(full_pipeline_graph, 'temperatures')
        assert hasattr(full_pipeline_graph, 'precipitation')
        
        # Verify data integrity
        assert len(full_pipeline_graph.temperatures) == len(full_pipeline_graph.points)
        assert len(full_pipeline_graph.precipitation) == len(full_pipeline_graph.points)
        
        # Verify reasonable ranges
        assert np.min(full_pipeline_graph.temperatures) >= -128
        assert np.max(full_pipeline_graph.temperatures) <= 127
        assert np.min(full_pipeline_graph.precipitation) >= 0
        assert np.max(full_pipeline_graph.precipitation) <= 255
        
    def test_climate_with_varied_terrain(self, full_pipeline_graph):
        """Test climate responds appropriately to terrain variation."""
        # Generate climate
        climate = Climate(full_pipeline_graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Find water and land cells
        water_cells = []
        land_cells = []
        mountain_cells = []
        
        for i in range(len(full_pipeline_graph.points)):
            if full_pipeline_graph.heights[i] < 20:
                water_cells.append(i)
            elif full_pipeline_graph.heights[i] > 60:
                mountain_cells.append(i)
            else:
                land_cells.append(i)
                
        # Mountains should be colder than lowlands (at same latitude)
        if mountain_cells and land_cells:
            # Find cells at similar latitudes
            mountain_temps = []
            land_temps = []
            
            for mountain_cell in mountain_cells[:10]:
                mountain_y = full_pipeline_graph.points[mountain_cell][1]
                mountain_temp = full_pipeline_graph.temperatures[mountain_cell]
                
                # Find nearby land cells
                for land_cell in land_cells:
                    land_y = full_pipeline_graph.points[land_cell][1]
                    if abs(mountain_y - land_y) < 20:  # Similar latitude
                        mountain_temps.append(mountain_temp)
                        land_temps.append(full_pipeline_graph.temperatures[land_cell])
                        break
                        
            if mountain_temps and land_temps:
                # Mountains should generally be colder
                assert np.mean(mountain_temps) < np.mean(land_temps)
                
    def test_climate_latitude_effects(self, full_pipeline_graph):
        """Test that climate varies appropriately with latitude."""
        # Use full latitude range
        map_coords = MapCoordinates(lat_n=90, lat_s=-90)
        climate = Climate(full_pipeline_graph, map_coords=map_coords)
        climate.calculate_temperatures()
        
        # Group cells by latitude bands
        north_temps = []
        equator_temps = []
        south_temps = []
        
        graph_height = full_pipeline_graph.graph_height
        
        for i in range(len(full_pipeline_graph.points)):
            y = full_pipeline_graph.points[i][1]
            latitude = 90 - (y / graph_height) * 180
            temp = full_pipeline_graph.temperatures[i]
            
            if latitude > 60:
                north_temps.append(temp)
            elif -20 < latitude < 20:
                equator_temps.append(temp)
            elif latitude < -60:
                south_temps.append(temp)
                
        # Equator should be warmer than poles
        if equator_temps and north_temps:
            assert np.mean(equator_temps) > np.mean(north_temps)
        if equator_temps and south_temps:
            assert np.mean(equator_temps) > np.mean(south_temps)
            
    def test_climate_coastal_effects(self, full_pipeline_graph):
        """Test that coastal areas receive appropriate precipitation."""
        climate = Climate(full_pipeline_graph)
        climate.generate_precipitation()
        
        # Find coastal cells (those near water)
        coastal_precip = []
        inland_precip = []
        
        for i in range(len(full_pipeline_graph.points)):
            if full_pipeline_graph.heights[i] >= 20:  # Land cell
                # Check if near water
                is_coastal = False
                if hasattr(full_pipeline_graph, 'distance_field'):
                    # Use distance field if available
                    is_coastal = abs(full_pipeline_graph.distance_field[i]) <= 2
                else:
                    # Check neighbors manually
                    for neighbor in full_pipeline_graph.cell_neighbors[i]:
                        if full_pipeline_graph.heights[neighbor] < 20:
                            is_coastal = True
                            break
                            
                if is_coastal:
                    coastal_precip.append(full_pipeline_graph.precipitation[i])
                else:
                    inland_precip.append(full_pipeline_graph.precipitation[i])
                    
        # Coastal areas should generally receive more precipitation
        if coastal_precip and inland_precip and len(coastal_precip) > 10 and len(inland_precip) > 10:
            # Allow some tolerance as this is a statistical effect
            coastal_avg = np.mean(coastal_precip)
            inland_avg = np.mean(inland_precip)
            
            # Coastal should have at least some precipitation advantage
            assert coastal_avg >= inland_avg * 0.8  # Allow 20% tolerance
            
    def test_climate_reproducibility(self, full_pipeline_graph):
        """Test that climate generation is reproducible with same parameters."""
        # Generate climate twice with same parameters
        climate1 = Climate(full_pipeline_graph)
        climate1.calculate_temperatures()
        climate1.generate_precipitation()
        
        temps1 = full_pipeline_graph.temperatures.copy()
        precip1 = full_pipeline_graph.precipitation.copy()
        
        # Reset and generate again
        full_pipeline_graph.temperatures = None
        full_pipeline_graph.precipitation = None
        
        climate2 = Climate(full_pipeline_graph)
        climate2.calculate_temperatures()
        climate2.generate_precipitation()
        
        temps2 = full_pipeline_graph.temperatures
        precip2 = full_pipeline_graph.precipitation
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(temps1, temps2)
        # Note: precipitation might have some randomness, so we test for high similarity
        correlation = np.corrcoef(precip1, precip2)[0, 1]
        assert correlation > 0.95  # Very high correlation expected
        
    def test_climate_data_persistence(self, full_pipeline_graph):
        """Test that climate data persists correctly on the graph."""
        # Generate climate
        climate = Climate(full_pipeline_graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Verify data is accessible through graph
        assert full_pipeline_graph.temperatures is not None
        assert full_pipeline_graph.precipitation is not None
        
        # Verify data types
        assert full_pipeline_graph.temperatures.dtype == np.int8
        assert full_pipeline_graph.precipitation.dtype == np.uint8
        
        # Verify no NaN or invalid values
        assert not np.any(np.isnan(full_pipeline_graph.temperatures.astype(float)))
        assert not np.any(np.isnan(full_pipeline_graph.precipitation.astype(float)))
        
    def test_climate_with_custom_options(self, full_pipeline_graph):
        """Test climate with custom options."""
        # Custom climate options
        custom_options = ClimateOptions(
            temperature_equator=30,
            temperature_north_pole=-40,
            precipitation_modifier=1.5
        )
        
        custom_coords = MapCoordinates(lat_n=60, lat_s=-60)
        
        climate = Climate(full_pipeline_graph, options=custom_options, map_coords=custom_coords)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Should complete without errors
        assert full_pipeline_graph.temperatures is not None
        assert full_pipeline_graph.precipitation is not None
        
        # Should use custom values
        assert climate.options.temperature_equator == 30
        assert climate.options.precipitation_modifier == 1.5
        assert climate.map_coords.lat_n == 60
        assert climate.map_coords.lat_s == -60


def test_climate_module_integration():
    """Test that climate module integrates properly with other modules."""
    from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates
    from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
    
    # Should be able to import and use together
    config = GridConfig(100, 100, 100)
    graph = generate_voronoi_graph(config, seed="integration_test")
    graph.heights = np.full(len(graph.points), 30, dtype=np.uint8)
    
    climate = Climate(graph)
    climate.calculate_temperatures()
    climate.generate_precipitation()
    
    # Should complete without errors
    assert graph.temperatures is not None
    assert graph.precipitation is not None

