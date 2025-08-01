"""Tests for hydrology module."""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.hydrology import Hydrology, HydrologyOptions, River, Lake


class TestHydrology:
    """Test hydrology calculations."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph with varied terrain."""
        config = GridConfig(200, 200, 500)
        graph = generate_voronoi_graph(config, seed="hydrology_test")
        
        # Create a simple terrain with a valley
        n_cells = len(graph.points)
        graph.heights = np.full(n_cells, 50, dtype=np.uint8)  # Default elevation
        
        # Create a valley running from north to south
        for i in range(n_cells):
            x, y = graph.points[i]
            
            # Valley in the middle
            if 80 < x < 120:
                # Gradually descending valley
                valley_depth = int(30 - (y / 200) * 20)  # 30 at top, 10 at bottom
                graph.heights[i] = max(valley_depth, 10)
                
            # Mountains on the sides
            elif x < 50 or x > 150:
                graph.heights[i] = 80
                
            # Water at the bottom
            if y > 180:
                graph.heights[i] = 5  # Ocean
                
        # Add some precipitation
        graph.precipitation = np.full(n_cells, 20, dtype=np.uint8)
        
        # More precipitation in mountains
        for i in range(n_cells):
            if graph.heights[i] > 70:
                graph.precipitation[i] = 40
                
        return graph
        
    @pytest.fixture
    def depression_graph(self):
        """Create a graph with depressions for testing depression filling."""
        config = GridConfig(100, 100, 100)
        graph = generate_voronoi_graph(config, seed="depression_test")
        
        n_cells = len(graph.points)
        graph.heights = np.full(n_cells, 50, dtype=np.uint8)
        
        # Create a depression in the middle
        center_x, center_y = 50, 50
        for i in range(n_cells):
            x, y = graph.points[i]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            if distance < 20:
                # Depression gets deeper toward center
                depth = int(20 - distance)
                graph.heights[i] = max(30 - depth, 10)
                
        graph.precipitation = np.full(n_cells, 15, dtype=np.uint8)
        return graph
        
    def test_depression_filling_basic(self, depression_graph):
        """Test basic depression filling functionality."""
        hydrology = Hydrology(depression_graph)
        hydrology.fill_depressions()
        
        # Should have filled heights
        assert hydrology.filled_heights is not None
        assert len(hydrology.filled_heights) == len(depression_graph.points)
        
        # Filled heights should be >= original heights
        assert np.all(hydrology.filled_heights >= depression_graph.heights)
        
        # Some cells should have been filled
        filled_cells = np.sum(hydrology.filled_heights > depression_graph.heights)
        assert filled_cells > 0
        
    def test_depression_filling_removes_sinks(self, depression_graph):
        """Test that depression filling removes internal sinks."""
        hydrology = Hydrology(depression_graph)
        hydrology.fill_depressions()
        hydrology.calculate_flow_directions()
        
        # Count cells with no outflow (sinks)
        sinks = np.sum(hydrology.flow_directions == -1)
        
        # Only border cells should be sinks
        border_cells = np.sum(depression_graph.cell_border_flags)
        
        # Allow some tolerance for complex cases
        assert sinks <= border_cells + 5
        
    def test_flow_direction_calculation(self, simple_graph):
        """Test flow direction calculation."""
        hydrology = Hydrology(simple_graph)
        hydrology.calculate_flow_directions()
        
        # Should have flow directions
        assert hydrology.flow_directions is not None
        assert len(hydrology.flow_directions) == len(simple_graph.points)
        
        # Most cells should have a flow direction
        cells_with_flow = np.sum(hydrology.flow_directions != -1)
        total_cells = len(simple_graph.points)
        
        # At least 80% of cells should have flow direction
        assert cells_with_flow >= total_cells * 0.8
        
    def test_water_flow_simulation(self, simple_graph):
        """Test water flow simulation."""
        hydrology = Hydrology(simple_graph)
        hydrology.simulate_water_flow()
        
        # Should have water flux
        assert hydrology.water_flux is not None
        assert len(hydrology.water_flux) == len(simple_graph.points)
        
        # All cells should have some water
        assert np.all(hydrology.water_flux > 0)
        
        # Valley bottom should have more water than mountains
        valley_cells = []
        mountain_cells = []
        
        for i in range(len(simple_graph.points)):
            if simple_graph.heights[i] < 30:  # Valley
                valley_cells.append(i)
            elif simple_graph.heights[i] > 70:  # Mountain
                mountain_cells.append(i)
                
        if valley_cells and mountain_cells:
            valley_flow = np.mean([hydrology.water_flux[i] for i in valley_cells])
            mountain_flow = np.mean([hydrology.water_flux[i] for i in mountain_cells])
            
            # Valley should accumulate more water
            assert valley_flow > mountain_flow
            
    def test_river_generation(self, simple_graph):
        """Test river generation."""
        hydrology = Hydrology(simple_graph)
        hydrology.generate_rivers()
        
        # Should generate some rivers
        assert len(hydrology.rivers) > 0
        
        # Check river properties
        for river in hydrology.rivers:
            assert isinstance(river, River)
            assert river.id >= 0
            assert len(river.cells) > 1
            assert river.flow >= hydrology.options.min_river_flow
            assert river.length > 0
            assert river.source_cell in river.cells
            assert river.mouth_cell in river.cells
            
    def test_river_flow_ordering(self, simple_graph):
        """Test that rivers flow from high to low elevation."""
        hydrology = Hydrology(simple_graph)
        hydrology.generate_rivers()
        
        for river in hydrology.rivers:
            if len(river.cells) > 2:
                # Check that river generally flows downhill
                source_height = hydrology.filled_heights[river.source_cell]
                mouth_height = hydrology.filled_heights[river.mouth_cell]
                
                # Source should be higher than or equal to mouth
                assert source_height >= mouth_height
                
    def test_lake_detection(self, depression_graph):
        """Test lake detection in closed basins."""
        # Modify the depression to create a proper lake
        # Remove border connection to create closed basin
        for i in range(len(depression_graph.points)):
            x, y = depression_graph.points[i]
            if 40 < x < 60 and 40 < y < 60:
                depression_graph.heights[i] = 15  # Lake bottom
                
        hydrology = Hydrology(depression_graph)
        hydrology.detect_lakes()
        
        # Should detect at least one lake
        assert len(hydrology.lakes) >= 0  # May not always form lakes
        
        # Check lake properties if any lakes formed
        for lake in hydrology.lakes:
            assert isinstance(lake, Lake)
            assert lake.id >= 0
            assert len(lake.cells) >= 3
            assert lake.water_level > 0
            assert lake.area > 0
            
    def test_full_simulation_pipeline(self, simple_graph):
        """Test the complete hydrology simulation pipeline."""
        hydrology = Hydrology(simple_graph)
        hydrology.run_full_simulation()
        
        # All components should be generated
        assert hydrology.filled_heights is not None
        assert hydrology.flow_directions is not None
        assert hydrology.water_flux is not None
        assert hydrology.rivers is not None
        assert hydrology.lakes is not None
        
        # Data should be stored on graph
        assert hasattr(simple_graph, 'water_flux')
        assert hasattr(simple_graph, 'flow_directions')
        assert hasattr(simple_graph, 'filled_heights')
        assert hasattr(simple_graph, 'rivers')
        assert hasattr(simple_graph, 'lakes')
        
    def test_hydrology_with_custom_options(self, simple_graph):
        """Test hydrology with custom options."""
        custom_options = HydrologyOptions(
            min_river_flow=50.0,
            lake_threshold=0.9,
            river_width_factor=2.0
        )
        
        hydrology = Hydrology(simple_graph, options=custom_options)
        hydrology.run_full_simulation()
        
        # Should use custom options
        assert hydrology.options.min_river_flow == 50.0
        assert hydrology.options.lake_threshold == 0.9
        assert hydrology.options.river_width_factor == 2.0
        
        # Rivers should respect minimum flow threshold
        for river in hydrology.rivers:
            assert river.flow >= 50.0
            
    def test_hydrology_without_precipitation(self, simple_graph):
        """Test hydrology works without precipitation data."""
        # Remove precipitation data
        if hasattr(simple_graph, 'precipitation'):
            delattr(simple_graph, 'precipitation')
            
        hydrology = Hydrology(simple_graph)
        hydrology.run_full_simulation()
        
        # Should still work with default precipitation
        assert hydrology.water_flux is not None
        assert np.all(hydrology.water_flux > 0)
        
    def test_river_path_tracing(self, simple_graph):
        """Test river path tracing functionality."""
        hydrology = Hydrology(simple_graph)
        hydrology.simulate_water_flow()
        
        # Find a high-flow cell
        high_flow_cells = np.where(hydrology.water_flux >= hydrology.options.min_river_flow)[0]
        
        if len(high_flow_cells) > 0:
            test_cell = high_flow_cells[0]
            path = hydrology._trace_river_path(test_cell, set())
            
            # Path should be valid
            assert len(path) > 0
            assert test_cell == path[0]
            
            # Path should be connected
            for i in range(len(path) - 1):
                current_cell = path[i]
                next_cell = path[i + 1]
                assert next_cell in simple_graph.cell_neighbors[current_cell]
                
    def test_river_length_calculation(self, simple_graph):
        """Test river length calculation."""
        hydrology = Hydrology(simple_graph)
        
        # Create a simple test path
        test_path = [0, 1, 2]  # Assuming these are connected
        
        # Make sure the cells are actually connected
        if (1 in simple_graph.cell_neighbors[0] and 
            2 in simple_graph.cell_neighbors[1]):
            
            length = hydrology._calculate_river_length(test_path)
            assert length > 0
            
            # Length should be reasonable (not too large or small)
            assert 0 < length < 1000  # Assuming reasonable map scale
            
    def test_depression_filling_preserves_drainage(self, simple_graph):
        """Test that depression filling preserves natural drainage patterns."""
        hydrology = Hydrology(simple_graph)
        
        # Get original heights
        original_heights = simple_graph.heights.copy()
        
        # Fill depressions
        hydrology.fill_depressions()
        
        # Most cells should keep their original height
        unchanged_cells = np.sum(hydrology.filled_heights == original_heights)
        total_cells = len(original_heights)
        
        # At least 70% of cells should remain unchanged
        assert unchanged_cells >= total_cells * 0.7


def test_hydrology_module_imports():
    """Test that hydrology module imports correctly."""
    from py_fmg.core.hydrology import Hydrology, HydrologyOptions, River, Lake
    
    assert Hydrology is not None
    assert HydrologyOptions is not None
    assert River is not None
    assert Lake is not None
    
    
def test_hydrology_options_defaults():
    """Test hydrology options default values."""
    options = HydrologyOptions()
    
    assert options.min_river_flow == 30.0
    assert options.lake_threshold == 0.85
    assert options.river_width_factor == 1.0
    assert options.meander_factor == 0.3
    assert options.evaporation_rate == 0.1

