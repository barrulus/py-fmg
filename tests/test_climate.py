"""Tests for climate calculation module."""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates


class TestClimate:
    """Test climate calculations."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        config = GridConfig(100, 100, 100)
        graph = generate_voronoi_graph(config, seed="climate_test")
        
        # Set some varied heights
        graph.heights[:] = 30  # Default land
        # Add some water
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            if x < 20:
                graph.heights[i] = 10  # Water
            elif x > 80:
                graph.heights[i] = 70  # Mountains
                
        return graph
        
    @pytest.fixture
    def equatorial_graph(self):
        """Create a graph centered on equator."""
        config = GridConfig(100, 100, 100)
        graph = generate_voronoi_graph(config, seed="equator_test")
        graph.heights[:] = 30  # All land
        return graph
        
    def test_temperature_calculation_basic(self, simple_graph):
        """Test basic temperature calculation."""
        climate = Climate(simple_graph)
        climate.calculate_temperatures()
        
        assert hasattr(simple_graph, 'temperatures')
        assert len(simple_graph.temperatures) == len(simple_graph.points)
        
        # Check temperature range
        assert np.min(simple_graph.temperatures) >= -128
        assert np.max(simple_graph.temperatures) <= 127
        
    def test_latitude_temperature_gradient(self, simple_graph):
        """Test that temperature decreases from equator to poles."""
        # Set map to span full latitude range
        climate = Climate(simple_graph, map_coords=MapCoordinates(lat_n=90, lat_s=-90))
        climate.calculate_temperatures()
        
        # Get temperatures at different latitudes
        temps_by_latitude = {}
        for i in range(len(simple_graph.points)):
            y = simple_graph.points[i][1]
            lat = 90 - (y / simple_graph.graph_height) * 180
            lat_band = int(lat / 30) * 30  # Group into 30° bands
            
            if lat_band not in temps_by_latitude:
                temps_by_latitude[lat_band] = []
            temps_by_latitude[lat_band].append(simple_graph.temperatures[i])
            
        # Average temperatures should decrease from equator to poles
        avg_temps = {lat: np.mean(temps) for lat, temps in temps_by_latitude.items()}
        
        # Check general trend (allowing for some variation)
        if 0 in avg_temps and 60 in avg_temps:
            assert avg_temps[0] > avg_temps[60]  # Equator warmer than 60°N
        if 0 in avg_temps and -60 in avg_temps:
            assert avg_temps[0] > avg_temps[-60]  # Equator warmer than 60°S
            
    def test_altitude_temperature_drop(self, simple_graph):
        """Test that higher altitude means lower temperature."""
        climate = Climate(simple_graph)
        climate.calculate_temperatures()
        
        # Find cells at same latitude but different heights
        low_cells = []
        high_cells = []
        
        for i in range(len(simple_graph.points)):
            y = simple_graph.points[i][1]
            if 45 < y < 55:  # Middle latitude band
                if simple_graph.heights[i] < 30:
                    low_cells.append(i)
                elif simple_graph.heights[i] > 60:
                    high_cells.append(i)
                    
        if low_cells and high_cells:
            avg_temp_low = np.mean([simple_graph.temperatures[i] for i in low_cells])
            avg_temp_high = np.mean([simple_graph.temperatures[i] for i in high_cells])
            
            # Higher altitude should be colder
            assert avg_temp_high < avg_temp_low
            
    def test_tropical_zone_temperatures(self, equatorial_graph):
        """Test tropical zone has minimal temperature variation."""
        # Set map to tropical zone
        climate = Climate(equatorial_graph, map_coords=MapCoordinates(lat_n=20, lat_s=-20))
        climate.calculate_temperatures()
        
        # All cells should have similar temperatures (since all same height)
        temp_range = np.max(equatorial_graph.temperatures) - np.min(equatorial_graph.temperatures)
        
        # Tropical zone should have small temperature range
        assert temp_range < 10  # Less than 10°C variation
        
    def test_precipitation_generation_basic(self, simple_graph):
        """Test basic precipitation generation."""
        climate = Climate(simple_graph)
        climate.generate_precipitation()
        
        assert hasattr(simple_graph, 'precipitation')
        assert len(simple_graph.precipitation) == len(simple_graph.points)
        
        # Check precipitation range
        assert np.min(simple_graph.precipitation) >= 0
        assert np.max(simple_graph.precipitation) <= 255
        
        # Should have some precipitation
        assert np.sum(simple_graph.precipitation) > 0
        
    def test_orographic_precipitation(self, simple_graph):
        """Test that mountains receive more precipitation on windward side."""
        climate = Climate(simple_graph)
        
        # Set westerly winds
        climate.options.winds = [135] * 6  # All westerly
        climate.generate_precipitation()
        
        # Find mountain cells and their western neighbors
        windward_precip = []
        leeward_precip = []
        
        for i in range(len(simple_graph.points)):
            if simple_graph.heights[i] > 60:  # Mountain
                x, y = simple_graph.points[i]
                
                # Check western (windward) and eastern (leeward) neighbors
                for j in range(len(simple_graph.points)):
                    x2, y2 = simple_graph.points[j]
                    if abs(y - y2) < 10:  # Same latitude band
                        if x2 < x - 5 and x2 > x - 15:  # Windward side
                            windward_precip.append(simple_graph.precipitation[j])
                        elif x2 > x + 5 and x2 < x + 15:  # Leeward side
                            leeward_precip.append(simple_graph.precipitation[j])
                            
        if windward_precip and leeward_precip:
            # Windward side should get more precipitation (orographic effect)
            assert np.mean(windward_precip) > np.mean(leeward_precip)
            
    def test_water_increases_humidity(self, simple_graph):
        """Test that wind passing over water gains humidity."""
        climate = Climate(simple_graph)
        climate.generate_precipitation()
        
        # Cells downwind of water should get precipitation
        water_cells = [i for i in range(len(simple_graph.points)) if simple_graph.heights[i] < 20]
        
        if water_cells:
            # Check precipitation near water
            near_water_precip = []
            for water_cell in water_cells[:5]:
                for neighbor in simple_graph.cell_neighbors[water_cell]:
                    if simple_graph.heights[neighbor] >= 20:  # Land neighbor
                        near_water_precip.append(simple_graph.precipitation[neighbor])
                        
            if near_water_precip:
                # Coastal areas should receive precipitation
                assert np.mean(near_water_precip) > 0
                
    def test_climate_with_temperature_interaction(self, simple_graph):
        """Test that very cold areas (permafrost) don't affect wind flow."""
        climate = Climate(simple_graph)
        
        # First calculate temperatures
        climate.calculate_temperatures()
        
        # Make some cells very cold
        for i in range(10):
            simple_graph.temperatures[i] = -10
            
        # Generate precipitation
        climate.generate_precipitation()
        
        # Should still complete without errors
        assert np.sum(simple_graph.precipitation) > 0
        
    def test_custom_climate_options(self, simple_graph):
        """Test climate with custom options."""
        options = ClimateOptions(
            temperature_equator=30,
            temperature_north_pole=-40,
            temperature_south_pole=-35,
            height_exponent=2.0,
            precipitation_modifier=1.5
        )
        
        climate = Climate(simple_graph, options=options)
        climate.calculate_temperatures()
        climate.generate_precipitation()
        
        # Should use custom values
        assert climate.options.temperature_equator == 30
        assert climate.options.precipitation_modifier == 1.5
        
        # Results should be generated
        assert simple_graph.temperatures is not None
        assert simple_graph.precipitation is not None
        
    def test_wind_patterns_by_latitude(self, simple_graph):
        """Test that different latitudes have different wind patterns."""
        # Full latitude range
        climate = Climate(simple_graph, map_coords=MapCoordinates(lat_n=90, lat_s=-90))
        
        # Check wind direction logic
        # Tier 0 (polar): easterly (225°)
        is_west, is_east, is_north, is_south = climate._get_wind_directions(0)
        assert is_east and not is_west
        
        # Tier 1 (temperate): westerly (135°)
        is_west, is_east, is_north, is_south = climate._get_wind_directions(1)
        assert is_west and not is_east
        
        # Tier 2 (tropical): trade winds (225°)
        is_west, is_east, is_north, is_south = climate._get_wind_directions(2)
        assert is_east and not is_west


def test_module_imports():
    """Test that climate module imports correctly."""
    from py_fmg.core.climate import Climate, ClimateOptions, MapCoordinates
    
    assert Climate is not None
    assert ClimateOptions is not None
    assert MapCoordinates is not None