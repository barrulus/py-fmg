"""
Unit tests for the hydrology system.

Tests cover:
- Height alteration for water flow
- Depression resolution algorithm
- Water drainage simulation
- River formation and properties
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from py_fmg.core.hydrology import Hydrology, HydrologyOptions, RiverData
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.features import Features, Feature


class TestHydrologyOptions:
    """Test hydrology options configuration."""
    
    def test_default_options(self):
        """Test default option values."""
        options = HydrologyOptions()
        assert options.sea_level == 20
        assert options.min_river_flux == 30.0
        assert options.max_depression_iterations == 100
        assert options.lake_elevation_increment == 0.2
        assert options.depression_elevation_increment == 0.1
        assert options.meandering_factor == 0.5
        assert options.width_scale_factor == 1.0
    
    def test_custom_options(self):
        """Test custom option values."""
        options = HydrologyOptions(
            sea_level=15,
            min_river_flux=25.0,
            max_depression_iterations=50
        )
        assert options.sea_level == 15
        assert options.min_river_flux == 25.0
        assert options.max_depression_iterations == 50


class TestRiverData:
    """Test river data structure."""
    
    def test_river_data_creation(self):
        """Test river data initialization."""
        river = RiverData(id=1)
        assert river.id == 1
        assert river.cells == []
        assert river.parent_id is None
        assert river.discharge == 0.0
        assert river.width == 0.0
        assert river.length == 0.0
        assert river.source_distance == 0.0
    
    def test_river_data_with_values(self):
        """Test river data with populated values."""
        river = RiverData(
            id=5,
            cells=[1, 2, 3],
            parent_id=2,
            discharge=150.0,
            width=5.5,
            length=1000.0
        )
        assert river.id == 5
        assert river.cells == [1, 2, 3]
        assert river.parent_id == 2
        assert river.discharge == 150.0
        assert river.width == 5.5
        assert river.length == 1000.0


class TestHydrologyInit:
    """Test hydrology initialization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create minimal test graph
        self.config = GridConfig(50, 50, 100)
        self.graph = generate_voronoi_graph(self.config, seed="test_hydrology")
        
        # Create mock features and climate
        self.features = Mock()
        self.features.detected_features = []
        
        self.climate = Mock()
        self.climate.precipitation = {i: 50.0 for i in range(len(self.graph.points))}
    
    def test_hydrology_initialization(self):
        """Test hydrology system initialization."""
        hydrology = Hydrology(self.graph, self.features, self.climate)
        
        assert hydrology.graph == self.graph
        assert hydrology.features == self.features
        assert hydrology.climate == self.climate
        assert isinstance(hydrology.options, HydrologyOptions)
        
        # Check array initialization
        assert len(hydrology.flux) == len(self.graph.points)
        assert len(hydrology.river_ids) == len(self.graph.points)
        assert len(hydrology.confluences) == len(self.graph.points)
        
        # Check initial values
        assert np.all(hydrology.flux == 0.0)
        assert np.all(hydrology.river_ids == 0)
        assert np.all(hydrology.confluences == False)
        
        # Check river tracking
        assert hydrology.rivers == {}
        assert hydrology.next_river_id == 1
    
    def test_hydrology_with_custom_options(self):
        """Test hydrology initialization with custom options."""
        options = HydrologyOptions(sea_level=15, min_river_flux=25.0)
        hydrology = Hydrology(self.graph, self.features, self.climate, options)
        
        assert hydrology.options.sea_level == 15
        assert hydrology.options.min_river_flux == 25.0


class TestHeightAlteration:
    """Test height alteration algorithm."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(30, 30, 50)
        self.graph = generate_voronoi_graph(self.config, seed="test_heights")
        
        # Set some heights above sea level
        for i in range(len(self.graph.heights)):
            self.graph.heights[i] = 25 + (i % 20)  # Heights from 25-44
        
        self.features = Mock()
        self.features.features = []  # Empty list for mock
        self.climate = Mock()
        self.climate.precipitation = {}
        
        self.hydrology = Hydrology(self.graph, self.features, self.climate)
    
    def test_alter_heights_stores_original(self):
        """Test that original heights are stored."""
        original = self.graph.heights.copy()
        self.hydrology.alter_heights()
        
        assert self.hydrology.original_heights is not None
        np.testing.assert_array_equal(self.hydrology.original_heights, original)
    
    def test_alter_heights_adds_variation(self):
        """Test that small variations are added to break ties."""
        original = self.graph.heights.copy()
        
        # Convert to float to allow small variations
        original_float = original.astype(np.float32)
        self.graph.heights = original_float.copy()
        
        self.hydrology.alter_heights()
        
        # Heights should be slightly different (but close) for float arrays
        differences = np.abs(self.graph.heights - original_float)
        
        # Only land cells should be modified  
        land_cells = original_float >= self.hydrology.options.sea_level
        
        # Most land cells should be modified (some might get zero variation by chance)
        modified_cells = differences[land_cells] > 0
        assert np.sum(modified_cells) >= len(modified_cells) * 0.8  # At least 80% modified
        
        # All variations should be small
        assert np.all(differences[land_cells] < 0.001)  # Small variations only
    
    def test_alter_heights_preserves_water(self):
        """Test that water cells are not modified."""
        # Set some cells as water
        for i in range(0, 10):
            self.graph.heights[i] = 10  # Below sea level
        
        original = self.graph.heights.copy()
        self.hydrology.alter_heights()
        
        # Water cells should be unchanged
        water_cells = original < self.hydrology.options.sea_level
        np.testing.assert_array_equal(
            self.graph.heights[water_cells],
            original[water_cells]
        )


class TestDepressionResolution:
    """Test depression resolution algorithm.""" 
    
    def setup_method(self):
        """Setup test fixtures with known depressions."""
        self.config = GridConfig(30, 30, 50)
        self.graph = generate_voronoi_graph(self.config, seed="test_depressions")
        
        # Create a known depression pattern
        for i, height in enumerate(self.graph.heights):
            if i < 5:
                self.graph.heights[i] = 30  # Depression center
            elif i < 15:
                self.graph.heights[i] = 35  # Surrounding higher ground
            else:
                self.graph.heights[i] = 25  # Regular terrain
        
        self.features = Mock()
        self.features.features = []  # Empty list for mock
        self.climate = Mock()
        
        self.hydrology = Hydrology(self.graph, self.features, self.climate)
    
    def test_is_depressed_detection(self):
        """Test depression detection logic."""
        # Create simple test case
        test_graph = Mock()
        test_graph.heights = np.array([25, 30, 30, 30])  # Cell 0 is depressed
        test_graph.cell_neighbors = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]  # Cell connectivity
        test_graph.points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Mock points for array size
        
        hydrology = Hydrology(test_graph, Mock(), Mock())
        
        # Cell 0 should be depressed (lower than neighbors 1,2,3)
        assert hydrology._is_depressed(0)
        
        # Cell 1 should not be depressed (same height as some neighbors)
        assert not hydrology._is_depressed(1)
    
    def test_get_neighbors(self):
        """Test neighbor finding from cell connectivity."""
        # Mock cell neighbors with known connectivity
        self.graph.cell_neighbors = [
            [1, 2, 3],      # Cell 0 neighbors
            [0, 2, 4],      # Cell 1 neighbors  
            [0, 1, 3, 4],   # Cell 2 neighbors
            [0, 2],         # Cell 3 neighbors
            [1, 2],         # Cell 4 neighbors
        ]
        
        neighbors = self.hydrology._get_neighbors(0)
        expected_neighbors = [1, 2, 3]
        assert neighbors == expected_neighbors
        
        neighbors = self.hydrology._get_neighbors(2)
        expected_neighbors = [0, 1, 3, 4]
        assert neighbors == expected_neighbors
    
    def test_get_min_neighbor_height(self):
        """Test minimum neighbor height calculation."""
        # Setup known neighbors
        self.graph.cell_neighbors = [[1, 2], [0], [0]]  # Cell 0 has neighbors 1,2
        
        # Heights: cell 0 has neighbors 1,2 with heights 30,35
        self.graph.heights = np.array([25, 30, 35])
        
        min_height = self.hydrology._get_min_neighbor_height(0)
        assert min_height == 30  # Minimum of neighbors 1(30) and 2(35)
    
    def test_resolve_depressions_eliminates_pits(self):
        """Test that depression resolution eliminates local minima."""
        # Create a clear depression
        for i in range(len(self.graph.heights)):
            if i == 0:
                self.graph.heights[i] = 20  # Depression
            else:
                self.graph.heights[i] = 30  # Higher surroundings
        
        # Run resolution
        self.hydrology.resolve_depressions()
        
        # Depression should be filled
        assert self.graph.heights[0] > 20  # Should be raised
    
    def test_resolve_depressions_max_iterations(self):
        """Test that resolution respects maximum iterations."""
        options = HydrologyOptions(max_depression_iterations=5)
        hydrology = Hydrology(self.graph, self.features, self.climate, options)
        
        # Should complete without error even with low iteration limit
        hydrology.resolve_depressions()  # Should not raise exception


class TestWaterDrainage:
    """Test water drainage simulation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(30, 30, 50)
        self.graph = generate_voronoi_graph(self.config, seed="test_drainage")
        
        # Set all heights above sea level
        self.graph.heights[:] = 25
        
        self.features = Mock()
        self.features.detected_features = []
        
        self.climate = Mock()
        self.climate.precipitation = {i: 40.0 for i in range(len(self.graph.points))}
        
        self.hydrology = Hydrology(self.graph, self.features, self.climate)
    
    def test_add_precipitation_flux(self):
        """Test precipitation flux addition."""
        self.hydrology._add_precipitation_flux()
        
        # All land cells should have flux > 0
        land_cells = self.graph.heights >= self.hydrology.options.sea_level
        assert np.all(self.hydrology.flux[land_cells] > 0)
        
        # Water cells should have flux = 0
        water_cells = self.graph.heights < self.hydrology.options.sea_level
        if np.any(water_cells):
            assert np.all(self.hydrology.flux[water_cells] == 0)
    
    def test_add_precipitation_flux_no_climate(self):
        """Test precipitation flux with missing climate data."""
        self.climate.precipitation = None
        del self.climate.precipitation  # Remove attribute entirely
        
        self.hydrology._add_precipitation_flux()
        
        # Should use default values
        land_cells = self.graph.heights >= self.hydrology.options.sea_level
        assert np.all(self.hydrology.flux[land_cells] > 0)
    
    def test_process_lake_drainage(self):
        """Test lake drainage processing."""
        # Create mock lake feature
        lake_feature = Mock()
        lake_feature.type = "lake"
        lake_feature.id = 1
        lake_feature.area = 100
        
        self.features.features = [lake_feature]
        
        # Mock feature_ids to map cells to lake
        self.features.feature_ids = [0, 1, 1, 1, 0]  # Cells 1,2,3 are lake
        
        # Set some flux in lake cells
        self.hydrology.flux[1] = 50.0
        self.hydrology.flux[2] = 50.0
        self.hydrology.flux[3] = 50.0
        
        self.hydrology._process_lake_drainage()
        
        # Should process without error
        # Detailed behavior depends on outlet finding logic
    
    def test_find_lake_outlet(self):
        """Test lake outlet finding."""
        # Create mock lake with cells
        lake_feature = Mock()
        lake_feature.id = 1
        lake_cells = [10, 11, 12]
        
        # Mock cell neighbors
        self.graph.cell_neighbors = [[] for _ in range(20)]  # Initialize empty neighbors
        self.graph.cell_neighbors[10] = [5, 11]  # Cell 10 has non-lake neighbor 5
        self.graph.cell_neighbors[11] = [6, 10, 12]  # Cell 11 has non-lake neighbor 6
        self.graph.cell_neighbors[12] = [7, 11]  # Cell 12 has non-lake neighbor 7
        
        # Mock feature_ids array - lake cells have id 1, others have id 0
        self.features.feature_ids = [0] * 20  # All cells initially non-lake
        self.features.feature_ids[10] = 1  # Lake cell
        self.features.feature_ids[11] = 1  # Lake cell
        self.features.feature_ids[12] = 1  # Lake cell
        
        # Set heights so cell 11 is lowest
        self.graph.heights[10] = 25
        self.graph.heights[11] = 23  # Lowest
        self.graph.heights[12] = 24
        
        outlet = self.hydrology._find_lake_outlet(lake_feature, lake_cells)
        assert outlet == 11  # Should find lowest perimeter cell
    
    def test_find_flow_target(self):
        """Test flow target identification."""
        # Mock cell neighbors with known connectivity
        self.graph.cell_neighbors = [[1, 2], [0], [0]]  # Cell 0 connected to 1,2
        
        # Set heights so water flows from 0 to 1 (lowest)
        self.graph.heights[0] = 30
        self.graph.heights[1] = 25  # Lowest neighbor
        self.graph.heights[2] = 28
        
        target = self.hydrology._find_flow_target(0)
        assert target == 1  # Should flow to lowest neighbor


class TestRiverFormation:
    """Test river formation and properties."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(20, 20, 30)
        self.graph = generate_voronoi_graph(self.config, seed="test_rivers")
        
        self.features = Mock()
        self.features.features = []  # Empty list for mock
        self.climate = Mock()
        
        self.hydrology = Hydrology(self.graph, self.features, self.climate)
    
    def test_create_new_river(self):
        """Test new river creation."""
        from_cell = 0
        to_cell = 1
        
        # Set sufficient flux
        self.hydrology.flux[from_cell] = 50.0
        
        self.hydrology._create_or_extend_river(from_cell, to_cell)
        
        # Should create new river
        assert len(self.hydrology.rivers) == 1
        river_id = self.hydrology.river_ids[from_cell]
        assert river_id > 0
        assert from_cell in self.hydrology.rivers[river_id].cells
    
    def test_extend_existing_river(self):
        """Test extending existing river."""
        # Create initial river
        river_id = 1
        self.hydrology.rivers[river_id] = RiverData(id=river_id, cells=[0])
        self.hydrology.river_ids[0] = river_id
        self.hydrology.next_river_id = 2
        
        # Extend river
        self.hydrology._create_or_extend_river(0, 1)
        
        # River should be extended
        assert 1 in self.hydrology.rivers[river_id].cells
        assert self.hydrology.river_ids[1] == river_id
    
    def test_river_confluence(self):
        """Test river confluence handling."""
        # Create two existing rivers
        river1_id = 1
        river2_id = 2
        
        self.hydrology.rivers[river1_id] = RiverData(id=river1_id, cells=[0])
        self.hydrology.rivers[river2_id] = RiverData(id=river2_id, cells=[1])
        self.hydrology.river_ids[0] = river1_id
        self.hydrology.river_ids[1] = river2_id
        
        # Set flux to determine dominance
        self.hydrology.flux[0] = 100.0  # Higher flux
        self.hydrology.flux[1] = 50.0   # Lower flux
        
        # Create confluence
        self.hydrology._create_or_extend_river(0, 1)
        
        # Should mark confluence
        assert self.hydrology.confluences[1] or self.hydrology.confluences[0]
    
    def test_calculate_river_width(self):
        """Test river width calculation."""
        # Test with different discharge values
        width1 = self.hydrology._calculate_river_width(100.0)
        width2 = self.hydrology._calculate_river_width(400.0)
        
        assert width1 > 0
        assert width2 > width1  # Higher discharge = wider river
        
        # Test minimum width
        width_small = self.hydrology._calculate_river_width(0.1)
        assert width_small >= 1.0  # Minimum width enforced
    
    def test_calculate_river_length(self):
        """Test river length calculation."""
        # Create river with known cell positions
        cells = [0, 1, 2]
        
        # Set known positions for distance calculation
        self.graph.points = np.array([
            [0, 0],    # Cell 0
            [10, 0],   # Cell 1 - 10 units away
            [20, 0],   # Cell 2 - 10 units away
        ])
        
        length = self.hydrology._calculate_river_length(cells)
        
        # Should be base distance * meandering factor
        expected_base = 20.0  # 10 + 10
        expected_meandered = expected_base * (1.0 + self.hydrology.options.meandering_factor)
        
        assert abs(length - expected_meandered) < 0.01
    
    def test_calculate_source_distance(self):
        """Test source to mouth distance calculation."""
        cells = [0, 1, 2]
        
        # Set positions
        self.graph.points = np.array([
            [0, 0],    # Source
            [5, 5],    # Middle
            [10, 0],   # Mouth
        ])
        
        distance = self.hydrology._calculate_source_distance(cells)
        expected = 10.0  # Direct distance from source to mouth
        
        assert abs(distance - expected) < 0.01


class TestHydrologyIntegration:
    """Integration tests for complete hydrology system."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.config = GridConfig(40, 40, 60)
        self.graph = generate_voronoi_graph(self.config, seed="test_integration")
        
        # Create realistic height distribution
        for i, (x, y) in enumerate(self.graph.points):
            # Create height gradient from center
            center_x, center_y = 20, 20
            distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
            self.graph.heights[i] = max(25 + (10 - distance) * 2, 10)
        
        # Create features
        self.features = Features(self.graph)
        self.features.markup_grid()
        
        # Create climate with precipitation
        self.climate = Mock()
        self.climate.precipitation = {i: 60.0 for i in range(len(self.graph.points))}
        
        self.hydrology = Hydrology(self.graph, self.features, self.climate)
    
    def test_full_river_generation(self):
        """Test complete river generation process."""
        rivers = self.hydrology.generate_rivers()
        
        # Should generate some rivers with sufficient precipitation
        assert isinstance(rivers, dict)
        
        # Each river should have valid properties
        for river_id, river in rivers.items():
            assert isinstance(river, RiverData)
            assert river.id == river_id
            assert len(river.cells) > 0
            assert river.width >= 0
            assert river.length >= 0
            assert river.discharge >= 0
    
    def test_river_network_connectivity(self):
        """Test that river network forms properly connected system."""
        self.hydrology.generate_rivers()
        
        # Check that rivers form connected network
        river_cells = set()
        for river in self.hydrology.rivers.values():
            river_cells.update(river.cells)
        
        # Rivers should exist
        if len(self.hydrology.rivers) > 0:
            assert len(river_cells) > 0
            
            # All river cells should have valid IDs
            for cell_id in river_cells:
                assert self.hydrology.river_ids[cell_id] > 0
    
    def test_water_conservation(self):
        """Test that water flow conserves mass."""
        self.hydrology.generate_rivers()
        
        # Total input should roughly equal total flux in system
        # (allowing for evaporation and other losses)
        total_input = sum(
            precip for precip in self.climate.precipitation.values()
            if precip > 0
        )
        
        total_flux = np.sum(self.hydrology.flux)
        
        # Should have reasonable relationship (not exact due to losses)
        assert total_flux >= 0
        if total_input > 0:
            assert total_flux <= total_input * 2  # Allow for accumulation effects


@pytest.fixture
def sample_hydrology():
    """Fixture providing a sample hydrology system for testing."""
    config = GridConfig(30, 30, 50)
    graph = generate_voronoi_graph(config, seed="sample")
    
    # Set heights above sea level
    graph.heights[:] = 25
    
    features = Mock()
    features.features = []  # Empty list for mock
    
    climate = Mock()
    climate.precipitation = {i: 50.0 for i in range(len(graph.points))}
    
    return Hydrology(graph, features, climate)


class TestHydrologyFixture:
    """Test using the sample hydrology fixture."""
    
    def test_fixture_initialization(self, sample_hydrology):
        """Test that fixture is properly initialized."""
        assert isinstance(sample_hydrology, Hydrology)
        assert len(sample_hydrology.flux) > 0
        assert len(sample_hydrology.river_ids) > 0
        assert sample_hydrology.next_river_id == 1
    
    def test_fixture_river_generation(self, sample_hydrology):
        """Test river generation with fixture."""
        rivers = sample_hydrology.generate_rivers()
        assert isinstance(rivers, dict)