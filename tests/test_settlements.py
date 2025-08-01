"""
Unit tests for the settlement and state generation system.

Tests cover:
- Cell suitability ranking
- Capital placement with spacing
- State creation and properties
- Town placement
- State expansion algorithm
- Settlement features and types
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from py_fmg.core.settlements import Settlements, SettlementOptions, Settlement, State
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph


class TestSettlementOptions:
    """Test settlement options configuration."""
    
    def test_default_options(self):
        """Test default option values."""
        options = SettlementOptions()
        assert options.states_number == 30
        assert options.manors_number == 1000
        assert options.growth_rate == 1.0
        assert options.states_growth_rate == 1.0
        assert options.size_variety == 3.0
        assert options.min_state_cells == 10
        assert options.capital_spacing_divisor == 2
        assert options.culture_same_bonus == -9
        assert options.culture_foreign_penalty == 100
    
    def test_custom_options(self):
        """Test custom option values."""
        options = SettlementOptions(
            states_number=20,
            manors_number=500,
            growth_rate=1.5
        )
        assert options.states_number == 20
        assert options.manors_number == 500
        assert options.growth_rate == 1.5


class TestSettlementData:
    """Test settlement data structures."""
    
    def test_settlement_creation(self):
        """Test settlement initialization."""
        settlement = Settlement(
            id=1,
            cell_id=100,
            x=50.5,
            y=75.2
        )
        assert settlement.id == 1
        assert settlement.cell_id == 100
        assert settlement.x == 50.5
        assert settlement.y == 75.2
        assert settlement.name == ""
        assert settlement.population == 0.0
        assert settlement.is_capital == False
        assert settlement.state_id == 0
        assert settlement.culture_id == 0
        assert settlement.port_id == 0
        assert settlement.type == "Generic"
    
    def test_capital_settlement(self):
        """Test capital settlement creation."""
        capital = Settlement(
            id=5,
            cell_id=200,
            x=100.0,
            y=100.0,
            name="Capital City",
            population=150.5,
            is_capital=True,
            state_id=1,
            culture_id=3
        )
        assert capital.is_capital == True
        assert capital.state_id == 1
        assert capital.population == 150.5
    
    def test_state_creation(self):
        """Test state initialization."""
        state = State(
            id=1,
            name="Test Kingdom",
            capital_id=5,
            culture_id=3
        )
        assert state.id == 1
        assert state.name == "Test Kingdom"
        assert state.capital_id == 5
        assert state.culture_id == 3
        assert state.expansionism == 1.0
        assert state.territory_cells == []
        assert state.removed == False
        assert state.locked == False


class TestSettlementsInit:
    """Test settlements system initialization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create minimal test graph
        self.config = GridConfig(50, 50, 100)
        self.graph = generate_voronoi_graph(self.config, seed="test_settlements")
        self.graph.width = 50
        self.graph.height = 50
        
        # Create mock features
        self.features = Mock()
        self.features.features = []
        self.features.feature_ids = np.zeros(len(self.graph.points), dtype=np.int32)
        
        # Create mock cultures
        self.cultures = Mock()
        self.cultures.cell_cultures = np.ones(len(self.graph.points), dtype=np.int32)
        self.cultures.cultures = {1: Mock(center=50)}
        
        # Create mock biomes
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes.get_habitability = Mock(return_value=100)
        self.biomes.get_expansion_cost = Mock(return_value=50)
    
    def test_settlements_initialization(self):
        """Test settlements system initialization."""
        settlements = Settlements(
            self.graph, 
            self.features, 
            self.cultures, 
            self.biomes
        )
        
        assert settlements.graph == self.graph
        assert settlements.features == self.features
        assert settlements.cultures == self.cultures
        assert settlements.biomes == self.biomes
        assert isinstance(settlements.options, SettlementOptions)
        
        # Check array initialization
        assert len(settlements.cell_suitability) == len(self.graph.points)
        assert len(settlements.cell_population) == len(self.graph.points)
        assert len(settlements.cell_state) == len(self.graph.points)
        assert len(settlements.cell_settlement) == len(self.graph.points)
        
        # Check initial values
        assert np.all(settlements.cell_suitability == 0)
        assert np.all(settlements.cell_population == 0.0)
        assert np.all(settlements.cell_state == 0)
        assert np.all(settlements.cell_settlement == 0)
        
        # Check data structures
        assert settlements.settlements == {}
        assert settlements.states == {}
        assert settlements.next_settlement_id == 1
        assert settlements.next_state_id == 1
    
    def test_settlements_with_custom_options(self):
        """Test settlements initialization with custom options."""
        options = SettlementOptions(states_number=15, growth_rate=2.0)
        settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes,
            options
        )
        
        assert settlements.options.states_number == 15
        assert settlements.options.growth_rate == 2.0


class TestCellRanking:
    """Test cell suitability ranking."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(30, 30, 50)
        self.graph = generate_voronoi_graph(self.config, seed="test_ranking")
        
        # Set some cells as land
        for i in range(len(self.graph.heights)):
            if i < 25:
                self.graph.heights[i] = 30  # Land
            else:
                self.graph.heights[i] = 10  # Water
        
        self.features = Mock()
        self.features.features = []
        self.cultures = Mock()
        self.cultures.cell_cultures = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes.get_habitability = Mock(return_value=100)
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes
        )
    
    def test_rank_cells_basic(self):
        """Test basic cell ranking functionality."""
        self.settlements.rank_cells()
        
        # Land cells should have positive suitability
        land_cells = self.graph.heights >= 20
        assert np.any(self.settlements.cell_suitability[land_cells] > 0)
        
        # Water cells should have zero suitability
        water_cells = self.graph.heights < 20
        assert np.all(self.settlements.cell_suitability[water_cells] == 0)
        assert np.all(self.settlements.cell_population[water_cells] == 0)
    
    def test_rank_cells_habitability(self):
        """Test biome habitability affects ranking."""
        # Mock different habitability values
        def mock_habitability(biome_id):
            return 50 if biome_id == 1 else 150
        
        self.biomes.get_habitability = Mock(side_effect=mock_habitability)
        self.biomes.cell_biomes[0] = 1  # Low habitability
        self.biomes.cell_biomes[1] = 2  # High habitability
        
        self.settlements.rank_cells()
        
        # Higher habitability should result in higher suitability
        assert self.settlements.cell_suitability[1] > self.settlements.cell_suitability[0]
    
    def test_rank_cells_elevation_penalty(self):
        """Test that high elevation reduces suitability."""
        # Set different elevations
        self.graph.heights[0] = 30  # Low elevation
        self.graph.heights[1] = 80  # High elevation
        
        self.settlements.rank_cells()
        
        # Lower elevation should have higher suitability
        assert self.settlements.cell_suitability[0] > self.settlements.cell_suitability[1]
    
    def test_rank_cells_river_bonus(self):
        """Test that rivers increase suitability."""
        # Add flux data for rivers
        self.graph.flux = np.zeros(len(self.graph.points))
        self.graph.confluences = np.zeros(len(self.graph.points))
        self.graph.flux[0] = 100  # River cell
        
        self.settlements.rank_cells()
        
        # River cell should have higher suitability than non-river
        assert self.settlements.cell_suitability[0] > self.settlements.cell_suitability[1]


class TestCapitalPlacement:
    """Test capital city placement algorithm."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(100, 100, 200)
        self.graph = generate_voronoi_graph(self.config, seed="test_capitals")
        self.graph.width = 100
        self.graph.height = 100
        
        # Create land area with varying suitability
        for i in range(len(self.graph.heights)):
            self.graph.heights[i] = 30 + (i % 20)  # Varying land heights
        
        self.features = Mock()
        self.features.features = []
        self.cultures = Mock()
        self.cultures.cell_cultures = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes.get_habitability = Mock(return_value=100)
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes
        )
        
        # Rank cells first
        self.settlements.rank_cells()
    
    def test_place_capitals_count(self):
        """Test correct number of capitals are placed."""
        self.settlements.options.states_number = 5
        capitals = self.settlements.place_capitals()
        
        assert len(capitals) == 5
        assert len(self.settlements.settlements) == 5
    
    def test_place_capitals_spacing(self):
        """Test capitals maintain minimum spacing."""
        self.settlements.options.states_number = 3
        capitals = self.settlements.place_capitals()
        
        # Check spacing between all capital pairs
        positions = [(c.x, c.y) for c in capitals]
        min_distance = float('inf')
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1, p2 = positions[i], positions[j]
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                min_distance = min(min_distance, distance)
        
        # Capitals should have meaningful spacing
        assert min_distance > 5.0  # Arbitrary minimum threshold
    
    def test_place_capitals_high_suitability(self):
        """Test capitals are placed in high suitability cells."""
        # Set specific high suitability cells
        high_suit_cells = [10, 50, 100]
        for cell in high_suit_cells:
            self.settlements.cell_suitability[cell] = 1000
        
        self.settlements.options.states_number = 3
        capitals = self.settlements.place_capitals()
        
        # At least some capitals should be in high suitability cells
        capital_cells = [c.cell_id for c in capitals]
        overlap = set(capital_cells) & set(high_suit_cells)
        assert len(overlap) > 0
    
    def test_place_capitals_culture_requirement(self):
        """Test capitals require cultured cells."""
        # Remove culture from some cells
        self.cultures.cell_cultures[100:150] = 0
        
        self.settlements.options.states_number = 5
        capitals = self.settlements.place_capitals()
        
        # All capitals should be in cultured cells
        for capital in capitals:
            assert self.cultures.cell_cultures[capital.cell_id] > 0


class TestStateCreation:
    """Test state creation from capitals."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(50, 50, 100)
        self.graph = generate_voronoi_graph(self.config, seed="test_states")
        
        self.features = Mock()
        self.cultures = Mock()
        self.cultures.cultures = {1: Mock(center=50), 2: Mock(center=75)}
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes
        )
        
        # Create test capitals
        self.capitals = [
            Settlement(id=1, cell_id=10, x=10, y=10, is_capital=True, culture_id=1),
            Settlement(id=2, cell_id=20, x=30, y=30, is_capital=True, culture_id=2)
        ]
        
        for cap in self.capitals:
            self.settlements.settlements[cap.id] = cap
    
    def test_create_states_from_capitals(self):
        """Test state creation from capital list."""
        self.settlements.create_states(self.capitals)
        
        # Should create neutral state + capital states
        assert len(self.settlements.states) == 3  # Neutral + 2 capitals
        assert 0 in self.settlements.states  # Neutral state
        assert self.settlements.states[0].name == "Neutrals"
        
        # Check capital states
        for i, capital in enumerate(self.capitals, 1):
            assert i in self.settlements.states
            state = self.settlements.states[i]
            assert state.capital_id == capital.id
            assert state.culture_id == capital.culture_id
            assert capital.state_id == i
    
    def test_state_expansionism(self):
        """Test state expansionism values."""
        # Set specific random seed for reproducibility
        np.random.seed(42)
        
        self.settlements.create_states(self.capitals)
        
        # Check expansionism is within expected range
        for state_id in range(1, len(self.settlements.states)):
            state = self.settlements.states[state_id]
            assert 1.0 <= state.expansionism <= 4.0  # 1 + (0 to 3)
    
    def test_state_center_cell(self):
        """Test states track their center cells."""
        self.settlements.create_states(self.capitals)
        
        for i, capital in enumerate(self.capitals, 1):
            state = self.settlements.states[i]
            assert state.center_cell == capital.cell_id


class TestTownPlacement:
    """Test secondary settlement placement."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(100, 100, 300)
        self.graph = generate_voronoi_graph(self.config, seed="test_towns")
        self.graph.width = 100
        self.graph.height = 100
        
        # Create varied terrain
        for i in range(len(self.graph.heights)):
            self.graph.heights[i] = 25 + (i % 30)
        
        self.features = Mock()
        self.features.features = []
        self.cultures = Mock()
        self.cultures.cell_cultures = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes.get_habitability = Mock(return_value=100)
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes
        )
        
        # Rank cells and place capitals first
        self.settlements.rank_cells()
        self.settlements.options.states_number = 3
        self.settlements.place_capitals()
    
    def test_place_towns_auto_count(self):
        """Test automatic town count calculation."""
        self.settlements.options.manors_number = 1000  # Auto mode
        self.settlements.place_towns()
        
        # Should place reasonable number of towns
        towns = [s for s in self.settlements.settlements.values() if not s.is_capital]
        assert len(towns) > 0
        assert len(towns) < len(self.graph.points) / 2  # Not too many
    
    def test_place_towns_fixed_count(self):
        """Test fixed town count placement."""
        self.settlements.options.manors_number = 10
        self.settlements.place_towns()
        
        # Should attempt to place requested number
        towns = [s for s in self.settlements.settlements.values() if not s.is_capital]
        assert len(towns) <= 10
    
    def test_town_spacing(self):
        """Test towns maintain spacing from other settlements."""
        self.settlements.options.manors_number = 5
        self.settlements.place_towns()
        
        # Get all settlement positions
        positions = [(s.x, s.y) for s in self.settlements.settlements.values()]
        
        # Check minimum spacing
        min_distance = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1, p2 = positions[i], positions[j]
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                min_distance = min(min_distance, distance)
        
        # Should have some minimum spacing
        assert min_distance > 0.5


class TestStateExpansion:
    """Test state territorial expansion algorithm."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(50, 50, 100)
        self.graph = generate_voronoi_graph(self.config, seed="test_expansion")
        
        # Create clear land area
        self.graph.heights[:] = 30
        
        self.features = Mock()
        self.features.features = []
        self.features.feature_ids = np.zeros(len(self.graph.points), dtype=np.int32)
        
        self.cultures = Mock()
        self.cultures.cell_cultures = np.ones(len(self.graph.points), dtype=np.int32)
        self.cultures.cultures = {1: Mock(center=25)}
        
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes.get_habitability = Mock(return_value=100)
        self.biomes.get_expansion_cost = Mock(return_value=50)
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes
        )
        
        # Create simple state setup
        self.settlements.rank_cells()
        capitals = [
            Settlement(id=1, cell_id=25, x=25, y=25, is_capital=True, culture_id=1)
        ]
        for cap in capitals:
            self.settlements.settlements[cap.id] = cap
            self.settlements.cell_settlement[cap.cell_id] = cap.id
        
        self.settlements.create_states(capitals)
    
    def test_expand_states_basic(self):
        """Test basic state expansion from capital."""
        self.settlements.expand_states()
        
        # Capital cell should belong to its state
        capital = self.settlements.settlements[1]
        assert self.settlements.cell_state[capital.cell_id] == 1
        
        # Some neighboring cells should be claimed
        claimed_cells = np.sum(self.settlements.cell_state > 0)
        assert claimed_cells > 1  # More than just capital
    
    def test_expand_states_culture_bonus(self):
        """Test same culture provides expansion bonus."""
        # Set different cultures
        self.cultures.cell_cultures[:50] = 1  # Same as state
        self.cultures.cell_cultures[50:] = 2  # Different culture
        
        self.settlements.expand_states()
        
        # More same-culture cells should be claimed
        same_culture_claimed = np.sum(
            (self.settlements.cell_state == 1) & 
            (self.cultures.cell_cultures == 1)
        )
        diff_culture_claimed = np.sum(
            (self.settlements.cell_state == 1) & 
            (self.cultures.cell_cultures == 2)
        )
        
        # Same culture should expand more easily
        if diff_culture_claimed > 0:
            ratio = same_culture_claimed / diff_culture_claimed
            assert ratio > 1.5  # Arbitrary threshold
    
    def test_expand_states_water_barrier(self):
        """Test water cells block expansion."""
        # Create water barrier
        self.graph.heights[40:45] = 10  # Water strip
        
        self.settlements.expand_states()
        
        # Cells beyond water should not be claimed
        beyond_water = self.settlements.cell_state[45:]
        assert np.all(beyond_water == 0)
    
    def test_expand_states_respects_locked(self):
        """Test expansion respects locked states."""
        # Create a locked state
        locked_state = State(
            id=2,
            name="Locked State",
            capital_id=2,
            culture_id=1,
            locked=True
        )
        self.settlements.states[2] = locked_state
        self.settlements.cell_state[30] = 2  # Assign a cell to locked state
        
        self.settlements.expand_states()
        
        # Locked state cell should not be overwritten
        assert self.settlements.cell_state[30] == 2


class TestSettlementSpecification:
    """Test settlement property calculation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GridConfig(50, 50, 100)
        self.graph = generate_voronoi_graph(self.config, seed="test_specify")
        
        # Add required graph attributes
        self.graph.cell_haven = np.zeros(len(self.graph.points), dtype=np.int32)
        self.graph.temperature = np.ones(len(self.graph.points)) * 10  # Above freezing
        self.graph.river_ids = np.zeros(len(self.graph.points), dtype=np.int32)
        self.graph.flux = np.zeros(len(self.graph.points))
        
        self.features = Mock()
        self.features.features = [Mock(id=0), Mock(id=1, cells=10, type="ocean")]
        
        self.cultures = Mock()
        self.biomes = Mock()
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes
        )
        
        # Rank cells
        self.settlements.rank_cells()
    
    def test_settlement_population(self):
        """Test population calculation."""
        # Create test settlement
        settlement = Settlement(
            id=1,
            cell_id=50,
            x=25,
            y=25,
            is_capital=False
        )
        self.settlements.settlements[1] = settlement
        self.settlements.cell_suitability[50] = 100
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.settlements.specify_settlements()
        
        # Should have calculated population
        assert settlement.population > 0
    
    def test_capital_population_bonus(self):
        """Test capitals get population bonus."""
        # Create capital and regular town
        capital = Settlement(id=1, cell_id=25, x=10, y=10, is_capital=True)
        town = Settlement(id=2, cell_id=26, x=15, y=15, is_capital=False)
        
        self.settlements.settlements[1] = capital
        self.settlements.settlements[2] = town
        self.settlements.cell_suitability[25] = 100
        self.settlements.cell_suitability[26] = 100
        
        # Set random seed
        np.random.seed(42)
        
        self.settlements.specify_settlements()
        
        # Capital should have higher population
        assert capital.population > town.population
    
    def test_port_status(self):
        """Test port status determination."""
        # Create coastal settlement
        settlement = Settlement(id=1, cell_id=30, x=20, y=20, is_capital=True)
        self.settlements.settlements[1] = settlement
        
        # Set up harbor access
        self.graph.cell_haven[30] = 1  # Points to feature 1
        self.graph.harbor_scores = np.ones(len(self.graph.points))
        
        self.settlements.specify_settlements()
        
        # Should be identified as port
        assert settlement.port_id == 1
    
    def test_settlement_type_highland(self):
        """Test highland settlement type."""
        settlement = Settlement(id=1, cell_id=40, x=30, y=30)
        self.settlements.settlements[1] = settlement
        
        # Set high elevation
        self.graph.heights[40] = 65
        
        self.settlements.specify_settlements()
        
        assert settlement.type == "Highland"
    
    def test_settlement_type_river(self):
        """Test river settlement type."""
        settlement = Settlement(id=1, cell_id=35, x=25, y=25)
        self.settlements.settlements[1] = settlement
        
        # Set river presence
        self.graph.river_ids[35] = 1
        self.graph.flux[35] = 150
        
        self.settlements.specify_settlements()
        
        assert settlement.type == "River"
    
    def test_settlement_features(self):
        """Test settlement feature assignment."""
        # Create settlements with different populations
        small = Settlement(id=1, cell_id=10, x=5, y=5)
        medium = Settlement(id=2, cell_id=20, x=15, y=15)
        large = Settlement(id=3, cell_id=30, x=25, y=25, is_capital=True)
        
        self.settlements.settlements[1] = small
        self.settlements.settlements[2] = medium
        self.settlements.settlements[3] = large
        
        # Manually set populations
        small.population = 5
        medium.population = 25
        large.population = 100
        
        # Set random seed
        np.random.seed(42)
        
        self.settlements.specify_settlements()
        
        # Larger settlements should have more features
        assert large.citadel == True  # Capital always has citadel
        assert large.walls == True  # Capital always has walls
        
        # Check feature probability increases with size
        large_features = sum([large.citadel, large.plaza, large.walls, large.temple])
        small_features = sum([small.citadel, small.plaza, small.walls, small.temple])
        assert large_features >= small_features


class TestSettlementIntegration:
    """Integration tests for complete settlement system."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.config = GridConfig(100, 100, 300)
        self.graph = generate_voronoi_graph(self.config, seed="test_integration")
        self.graph.width = 100
        self.graph.height = 100
        
        # Create realistic terrain
        for i, (x, y) in enumerate(self.graph.points):
            # Create elevation gradient
            distance = ((x - 50)**2 + (y - 50)**2)**0.5
            self.graph.heights[i] = max(25 + (25 - distance) * 2, 5)
        
        # Add graph attributes
        self.graph.cell_haven = np.zeros(len(self.graph.points), dtype=np.int32)
        self.graph.temperature = np.ones(len(self.graph.points)) * 15
        self.graph.river_ids = np.zeros(len(self.graph.points), dtype=np.int32)
        self.graph.flux = np.zeros(len(self.graph.points))
        self.graph.cell_areas = np.ones(len(self.graph.points))
        
        # Create features
        self.features = Mock()
        self.features.features = [Mock(id=0)]
        self.features.feature_ids = np.zeros(len(self.graph.points), dtype=np.int32)
        
        # Create cultures
        self.cultures = Mock()
        self.cultures.cell_cultures = np.zeros(len(self.graph.points), dtype=np.int32)
        # Assign cultures to land cells
        for i in range(len(self.graph.points)):
            if self.graph.heights[i] >= 20:
                self.cultures.cell_cultures[i] = 1 + (i % 3)  # 3 cultures
        self.cultures.cultures = {
            1: Mock(center=50),
            2: Mock(center=100),
            3: Mock(center=150)
        }
        
        # Create biomes
        self.biomes = Mock()
        self.biomes.cell_biomes = np.ones(len(self.graph.points), dtype=np.int32)
        self.biomes.get_habitability = Mock(return_value=100)
        self.biomes.get_expansion_cost = Mock(return_value=50)
        
        options = SettlementOptions(
            states_number=5,
            manors_number=20
        )
        
        self.settlements = Settlements(
            self.graph,
            self.features,
            self.cultures,
            self.biomes,
            options
        )
    
    def test_full_generation(self):
        """Test complete settlement generation process."""
        settlements, states = self.settlements.generate()
        
        # Should generate requested states
        assert len(states) == 6  # Including neutral
        assert len([s for s in settlements.values() if s.is_capital]) == 5
        
        # Should generate towns
        towns = [s for s in settlements.values() if not s.is_capital]
        assert len(towns) > 0
        
        # All settlements should have properties
        for settlement in settlements.values():
            assert settlement.population > 0
            assert settlement.type != ""
            assert settlement.state_id >= 0
    
    def test_state_territory_assignment(self):
        """Test states claim territory."""
        self.settlements.generate()
        
        # Some cells should be assigned to states
        claimed_cells = np.sum(self.settlements.cell_state > 0)
        land_cells = np.sum(self.graph.heights >= 20)
        
        # Significant portion of land should be claimed
        assert claimed_cells > land_cells * 0.3
        
        # Each state should have some territory
        for state_id in range(1, 6):
            state_cells = np.sum(self.settlements.cell_state == state_id)
            assert state_cells > 0
    
    def test_cultural_clustering(self):
        """Test states tend to form along cultural lines."""
        self.settlements.generate()
        
        # Check cultural coherence of states
        for state_id in range(1, 6):
            state = self.settlements.states[state_id]
            state_cells = np.where(self.settlements.cell_state == state_id)[0]
            
            if len(state_cells) > 0:
                # Get cultures in state territory
                state_cultures = self.cultures.cell_cultures[state_cells]
                # Most common culture should match state culture
                if len(state_cultures) > 0:
                    most_common = np.bincount(state_cultures).argmax()
                    assert most_common == state.culture_id or len(state_cells) < 5
    
    def test_settlement_state_consistency(self):
        """Test settlements belong to states in their territory."""
        self.settlements.generate()
        
        for settlement in self.settlements.settlements.values():
            if settlement.id == 0:  # Skip placeholder
                continue
                
            cell_state = self.settlements.cell_state[settlement.cell_id]
            # Settlement state should match cell state
            assert settlement.state_id == cell_state


@pytest.fixture
def sample_settlements():
    """Fixture providing a sample settlement system for testing."""
    config = GridConfig(50, 50, 100)
    graph = generate_voronoi_graph(config, seed="sample")
    graph.width = 50
    graph.height = 50
    
    # Set heights above sea level
    graph.heights[:] = 30
    
    features = Mock()
    features.features = []
    
    cultures = Mock()
    cultures.cell_cultures = np.ones(len(graph.points), dtype=np.int32)
    cultures.cultures = {1: Mock(center=50)}
    
    biomes = Mock()
    biomes.cell_biomes = np.ones(len(graph.points), dtype=np.int32)
    biomes.get_habitability = Mock(return_value=100)
    
    return Settlements(graph, features, cultures, biomes)


class TestSettlementsFixture:
    """Test using the sample settlements fixture."""
    
    def test_fixture_initialization(self, sample_settlements):
        """Test that fixture is properly initialized."""
        assert isinstance(sample_settlements, Settlements)
        assert len(sample_settlements.cell_suitability) > 0
        assert sample_settlements.next_settlement_id == 1
        assert sample_settlements.next_state_id == 1
    
    def test_fixture_generation(self, sample_settlements):
        """Test settlement generation with fixture."""
        settlements, states = sample_settlements.generate()
        assert isinstance(settlements, dict)
        assert isinstance(states, dict)
        assert len(states) > 0  # At least neutral state