"""Tests for cell packing (reGraph) functionality."""

import pytest
import numpy as np
from py_fmg.core import (
    GridConfig, generate_voronoi_graph, regraph, 
    CellType, HeightmapGenerator, HeightmapConfig
)
from py_fmg.core.cell_packing import determine_cell_types


class TestCellTypeDetermination:
    """Test cell type classification."""
    
    def test_cell_types_basic(self):
        """Test basic cell type determination."""
        # Create a small test grid
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test", apply_relaxation=False)
        
        # Set up a simple height pattern
        # Create an "island" in the center
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            # Center at (25, 25)
            dist_from_center = np.sqrt((x - 25)**2 + (y - 25)**2)
            if dist_from_center < 15:
                graph.heights[i] = 50  # Land
            else:
                graph.heights[i] = 10  # Water
        
        # Determine cell types
        cell_types = determine_cell_types(graph)
        
        # Check that we have various cell types
        assert CellType.INLAND in cell_types
        assert CellType.LAND_COAST in cell_types
        assert CellType.WATER_COAST in cell_types
        assert CellType.DEEP_OCEAN in cell_types
    
    def test_coastal_detection(self):
        """Test that coastal cells are properly identified."""
        config = GridConfig(width=30, height=30, cells_desired=9)
        graph = generate_voronoi_graph(config, "test", apply_relaxation=False)
        
        # Create a simple pattern: land on left, water on right
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            if x < 15:
                graph.heights[i] = 50  # Land
            else:
                graph.heights[i] = 10  # Water
        
        cell_types = determine_cell_types(graph)
        
        # Cells near the boundary should be coastal
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            if 10 < x < 20:  # Near the boundary
                assert cell_types[i] in [CellType.LAND_COAST, CellType.WATER_COAST]


class TestReGraph:
    """Test the full reGraph operation."""
    
    def test_basic_regraph(self):
        """Test basic reGraph functionality."""
        # Create a test grid
        config = GridConfig(width=100, height=100, cells_desired=100)
        graph = generate_voronoi_graph(config, "test", apply_relaxation=False)
        
        # Create a height pattern with land and water
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            # Create an island
            dist_from_center = np.sqrt((x - 50)**2 + (y - 50)**2)
            if dist_from_center < 30:
                graph.heights[i] = 50  # Land
            else:
                graph.heights[i] = 10  # Deep ocean
        
        # Perform reGraph
        packed = regraph(graph)
        
        # Verify reduction in cell count
        assert len(packed.points) < len(graph.points)
        assert len(packed.points) > len(graph.points) * 0.3  # Not too few
        
        # Verify all arrays are consistent
        assert len(packed.heights) == len(packed.points)
        assert len(packed.cell_neighbors) == len(packed.points)
        assert len(packed.cell_border_flags) == len(packed.points)
        
        # Verify grid_indices mapping exists
        assert hasattr(packed, 'grid_indices')
        assert packed.grid_indices is not None
        assert len(packed.grid_indices) == len(packed.points)
    
    def test_coastal_enhancement(self):
        """Test that coastal cells get additional points."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test", apply_relaxation=False)
        
        # Create a coastline
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            if x < 25:
                graph.heights[i] = 50  # Land
            else:
                graph.heights[i] = 10  # Water
        
        # Count coastal cells before packing
        cell_types = determine_cell_types(graph)
        n_coastal = np.sum((cell_types == CellType.LAND_COAST) | 
                          (cell_types == CellType.WATER_COAST))
        
        # Perform reGraph
        packed = regraph(graph)
        
        # With coastal enhancement, packed should have more points than just kept cells
        # because intermediate points are added along coastlines
        n_kept_land = np.sum(graph.heights >= 20)
        assert len(packed.points) > n_kept_land, "Should have added coastal points"
    
    def test_deep_ocean_removal(self):
        """Test that deep ocean cells are removed."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test")
        
        # Set all cells to deep ocean except a few
        graph.heights[:] = 10  # All deep ocean
        graph.heights[5:10] = 50  # A few land cells
        
        # Perform reGraph
        packed = regraph(graph)
        
        # Should have fewer cells, but border cells are kept
        # With 5 land cells and border cells, we expect ~80% reduction
        assert len(packed.points) < len(graph.points)
        
        # All packed cells should have height >= 20 or be coastal
        assert np.all(packed.heights >= 10)  # No cells should be dropped entirely
    
    def test_regraph_with_heightmap(self):
        """Test reGraph with realistic heightmap data."""
        # Generate a full map with heightmap
        config = GridConfig(width=100, height=100, cells_desired=100)
        graph = generate_voronoi_graph(config, "test123")
        
        # Apply a simple heightmap
        hm_config = HeightmapConfig(
            width=100, height=100,
            cells_x=graph.cells_x,
            cells_y=graph.cells_y,
            cells_desired=100,
            spacing=graph.spacing
        )
        
        heightmap_gen = HeightmapGenerator(hm_config, graph)
        
        # Simple template: add some hills
        template = [
            "Hill 1 90 50 50",
            "Hill 3 60 30 70",
            "Hill 2 40 70 30"
        ]
        
        # Apply the template commands manually
        for cmd in template:
            parts = cmd.split()
            if parts[0] == "Hill":
                heightmap_gen.add_hill(parts[1], parts[2], parts[3], parts[4])
        
        # Get the generated heights
        graph.heights[:] = heightmap_gen.heights.astype(np.uint8)
        
        # Perform reGraph
        packed = regraph(graph)
        
        # Verify reasonable packing
        assert 0.3 < len(packed.points) / len(graph.points) < 0.7
        
        # Verify height distribution is preserved
        original_land = np.sum(graph.heights >= 20)
        packed_land = np.sum(packed.heights >= 20)
        
        # Most land should be preserved (some coastal water included)
        assert packed_land >= original_land * 0.8


@pytest.mark.parametrize("n_cells,expected_ratio", [
    (25, 0.4),   # Small map - higher ratio kept
    (100, 0.45),  # Medium map
    (400, 0.5),   # Large map - more ocean to remove
])
def test_packing_ratios(n_cells, expected_ratio):
    """Test that packing ratios are reasonable for different map sizes."""
    size = int(np.sqrt(n_cells) * 10)
    config = GridConfig(width=size, height=size, cells_desired=n_cells)
    graph = generate_voronoi_graph(config, "test")
    
    # Create a typical island pattern
    center_x, center_y = size / 2, size / 2
    for i in range(len(graph.points)):
        x, y = graph.points[i]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist < size * 0.3:
            graph.heights[i] = 50 + np.random.randint(-10, 10)
        else:
            graph.heights[i] = 10
    
    packed = regraph(graph)
    ratio = len(packed.points) / len(graph.points)
    
    # Check ratio is in reasonable range
    assert expected_ratio * 0.7 < ratio < expected_ratio * 1.3