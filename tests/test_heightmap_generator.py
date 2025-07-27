"""
Tests for heightmap generation module.
"""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.config.heightmap_templates import get_template, list_templates


class TestHeightmapGenerator:
    """Test heightmap generation functionality."""
    
    @pytest.fixture
    def small_graph(self):
        """Create a small test graph."""
        config = GridConfig(width=100, height=100, cells_desired=100)
        return generate_voronoi_graph(config, seed="test123")
    
    @pytest.fixture
    def heightmap_config(self):
        """Create test heightmap configuration."""
        return HeightmapConfig(
            width=100,
            height=100,
            cells_x=10,
            cells_y=10,
            cells_desired=100,
            spacing=10.0
        )
    
    def test_heightmap_initialization(self, small_graph, heightmap_config):
        """Test heightmap generator initialization."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        assert generator.heights.shape == (len(small_graph.points),)
        assert generator.heights.dtype == np.uint8
        assert np.all(generator.heights == 0)
        assert generator.blob_power > 0
        assert generator.line_power > 0
    
    def test_add_single_hill(self, small_graph, heightmap_config):
        """Test adding a single hill."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Add a hill in the center
        generator.add_hill(1, 50, "40-60", "40-60")
        
        # Check that heights were modified
        assert np.any(generator.heights > 0)
        assert np.max(generator.heights) <= 100
        
        # Check that hill spreads from center
        center_cells = []
        for i, point in enumerate(small_graph.points):
            if 40 <= point[0] <= 60 and 40 <= point[1] <= 60:
                center_cells.append(i)
        
        if center_cells:
            avg_center = np.mean([generator.heights[i] for i in center_cells])
            avg_total = np.mean(generator.heights)
            assert avg_center > avg_total  # Center should be higher
    
    def test_add_pit(self, small_graph, heightmap_config):
        """Test adding a pit (depression)."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Start with some elevation
        generator.heights[:] = 50
        
        # Add a pit
        generator.add_pit(1, 30, "40-60", "40-60")
        
        # Check that heights were reduced
        assert np.any(generator.heights < 50)
        assert np.min(generator.heights) >= 0
    
    def test_add_range(self, small_graph, heightmap_config):
        """Test adding a mountain range."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Add a range
        generator.add_range(1, 40, "10-20", "10-20")
        
        # Check that heights were modified
        assert np.any(generator.heights > 0)
        
        # Range should create a line of elevated cells
        elevated_count = np.sum(generator.heights > 20)
        assert elevated_count >= 3  # Should affect multiple cells
    
    def test_smooth_operation(self, small_graph, heightmap_config):
        """Test smoothing operation."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Create some random heights
        generator.heights = np.random.randint(0, 100, size=len(small_graph.points), dtype=np.uint8)
        original_std = np.std(generator.heights)
        
        # Apply smoothing
        generator.smooth(factor=2)
        
        # Standard deviation should decrease (heights more uniform)
        new_std = np.std(generator.heights)
        assert new_std < original_std
    
    def test_mask_operation(self, small_graph, heightmap_config):
        """Test mask operation (edge fading)."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Set uniform heights
        generator.heights[:] = 80
        
        # Apply mask
        generator.mask(power=1)
        
        # Check that edges are lower than center
        edge_cells = []
        center_cells = []
        
        for i, point in enumerate(small_graph.points):
            x, y = point
            if x < 10 or x > 90 or y < 10 or y > 90:
                edge_cells.append(i)
            elif 40 <= x <= 60 and 40 <= y <= 60:
                center_cells.append(i)
        
        if edge_cells and center_cells:
            avg_edge = np.mean([generator.heights[i] for i in edge_cells])
            avg_center = np.mean([generator.heights[i] for i in center_cells])
            assert avg_edge < avg_center
    
    def test_modify_operation(self, small_graph, heightmap_config):
        """Test modify operation."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Set some test heights
        generator.heights[:50] = 30  # Land
        generator.heights[50:] = 10   # Water
        
        # Modify only land cells
        generator.modify("land", add=10)
        
        # Check modifications
        assert np.all(generator.heights[:50] >= 30)  # Land cells increased
        assert np.all(generator.heights[50:] == 10)  # Water cells unchanged
    
    def test_strait_operation(self, small_graph, heightmap_config):
        """Test strait (water channel) creation."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        # Start with elevated terrain
        generator.heights[:] = 60
        
        # Add a strait
        generator.add_strait(2, "vertical")
        
        # Check that some cells were lowered
        assert np.any(generator.heights < 60)
        assert np.min(generator.heights) < 20  # Should create water
    
    def test_template_volcano(self, small_graph, heightmap_config):
        """Test volcano template generation."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        template = get_template("volcano")
        heights = generator.from_template(template, seed="volcano123")
        
        # Volcano should have high peak
        assert np.max(heights) > 70
        # And surrounding lower areas
        assert np.min(heights) < 30
        # With reasonable variation
        assert 10 < np.std(heights) < 40
    
    def test_template_archipelago(self, small_graph, heightmap_config):
        """Test archipelago template generation."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        template = get_template("archipelago")
        heights = generator.from_template(template, seed="arch123")
        
        # Archipelago should have mix of land and water
        land_cells = np.sum(heights >= 20)
        water_cells = np.sum(heights < 20)
        
        assert land_cells > 5  # Some land
        assert water_cells > 5  # Some water
        # Archipelago template creates mostly water with small islands
        assert 0.05 < land_cells / len(heights) < 0.5  # Islands in water
    
    def test_reproducibility(self, small_graph, heightmap_config):
        """Test that same seed produces same results."""
        generator1 = HeightmapGenerator(heightmap_config, small_graph)
        generator2 = HeightmapGenerator(heightmap_config, small_graph)
        
        template = get_template("continents")
        
        heights1 = generator1.from_template(template, seed="repro123")
        heights2 = generator2.from_template(template, seed="repro123")
        
        np.testing.assert_array_equal(heights1, heights2)
    
    def test_all_templates_valid(self, small_graph, heightmap_config):
        """Test that all templates can be generated without errors."""
        generator = HeightmapGenerator(heightmap_config, small_graph)
        
        for template_name in list_templates():
            template = get_template(template_name)
            heights = generator.from_template(template, seed=f"{template_name}123")
            
            # Basic validation
            assert heights.shape == (len(small_graph.points),)
            assert np.all(heights >= 0)
            assert np.all(heights <= 100)
            assert np.any(heights > 0)  # Not all zeros


class TestHeightmapBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def tiny_graph(self):
        """Create a very small test graph."""
        config = GridConfig(width=30, height=30, cells_desired=10)
        return generate_voronoi_graph(config, seed="tiny")
    
    def test_operations_on_tiny_map(self, tiny_graph):
        """Test that operations work on very small maps."""
        config = HeightmapConfig(
            width=30,
            height=30,
            cells_x=3,
            cells_y=3,
            cells_desired=10,
            spacing=10.0
        )
        
        generator = HeightmapGenerator(config, tiny_graph)
        
        # Test various operations
        generator.add_hill(1, 50, "30-70", "30-70")
        assert np.any(generator.heights > 0)
        
        generator.add_pit(1, 20, "40-60", "40-60")
        generator.smooth(2)
        generator.mask(1)
        
        # Should complete without errors
        assert generator.heights.shape == (len(tiny_graph.points),)
    
    def test_extreme_parameters(self, tiny_graph):
        """Test operations with extreme parameters."""
        config = HeightmapConfig(
            width=30,
            height=30,
            cells_x=3,
            cells_y=3,
            cells_desired=10,
            spacing=10.0
        )
        
        generator = HeightmapGenerator(config, tiny_graph)
        
        # Very high hill
        generator.add_hill(1, 100, "45-55", "45-55")
        assert np.all(generator.heights <= 100)
        
        # Very deep pit
        generator.heights[:] = 80
        generator.add_pit(1, 100, "45-55", "45-55")
        assert np.all(generator.heights >= 0)
        
        # Very wide strait
        generator.heights[:] = 50
        generator.add_strait(10, "vertical")  # Wider than map
        assert np.all(generator.heights >= 0)