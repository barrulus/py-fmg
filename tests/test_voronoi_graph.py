"""Tests for Voronoi graph generation."""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import (
    GridConfig, generate_voronoi_graph, get_jittered_grid, 
    get_boundary_points, find_grid_cell
)


class TestJitteredGrid:
    """Test jittered grid generation."""
    
    def test_grid_size(self):
        """Test that grid generates expected number of points."""
        width, height, spacing = 100, 100, 10
        points = get_jittered_grid(width, height, spacing, "test_seed")
        
        # Expected grid is roughly 10x10 = 100 points
        expected_count = (width // spacing) * (height // spacing)
        assert len(points) == expected_count
    
    def test_point_bounds(self):
        """Test that all points are within bounds."""
        width, height, spacing = 100, 100, 10
        points = get_jittered_grid(width, height, spacing, "test_seed")
        
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 0] <= width)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 1] <= height)
    
    def test_jittering_consistency(self):
        """Test that same seed produces same jittering."""
        width, height, spacing = 50, 50, 5
        points1 = get_jittered_grid(width, height, spacing, "test_seed")
        points2 = get_jittered_grid(width, height, spacing, "test_seed")
        
        np.testing.assert_array_equal(points1, points2)
    
    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        width, height, spacing = 50, 50, 5
        points1 = get_jittered_grid(width, height, spacing, "seed1")
        points2 = get_jittered_grid(width, height, spacing, "seed2")
        
        # Points should not be identical
        assert not np.array_equal(points1, points2)


class TestBoundaryPoints:
    """Test boundary point generation."""
    
    def test_boundary_count(self):
        """Test boundary point generation."""
        width, height, spacing = 100, 100, 10
        boundary = get_boundary_points(width, height, spacing)
        
        # Should have points along all 4 edges
        assert len(boundary) > 0
        assert boundary.shape[1] == 2  # [x, y] coordinates
    
    def test_boundary_placement(self):
        """Test that boundary points are outside main area."""
        width, height, spacing = 100, 100, 10
        boundary = get_boundary_points(width, height, spacing)
        
        # Boundary points should be at negative offset or beyond dimensions
        offset = -spacing
        extended_width = width - offset * 2
        extended_height = height - offset * 2
        
        # Check that some points are on boundaries
        on_left = np.any(boundary[:, 0] == offset)
        on_right = np.any(boundary[:, 0] == extended_width + offset)
        on_top = np.any(boundary[:, 1] == offset)
        on_bottom = np.any(boundary[:, 1] == extended_height + offset)
        
        assert on_left or on_right or on_top or on_bottom


class TestVoronoiGraph:
    """Test complete Voronoi graph generation."""
    
    def test_basic_generation(self):
        """Test basic graph generation."""
        config = GridConfig(width=100, height=100, cells_desired=100)
        graph = generate_voronoi_graph(config, "test_seed")
        
        assert graph.cells_desired == 100
        assert len(graph.points) > 0
        assert len(graph.cell_neighbors) == len(graph.points)
        assert len(graph.cell_vertices) == len(graph.points)
        assert len(graph.cell_border_flags) == len(graph.points)
    
    def test_connectivity_integrity(self):
        """Test that cell connectivity is symmetric."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test_seed")
        
        # Check that if cell A is neighbor of B, then B is neighbor of A
        for i, neighbors in enumerate(graph.cell_neighbors):
            for neighbor in neighbors:
                assert i in graph.cell_neighbors[neighbor], \
                    f"Cell {i} lists {neighbor} as neighbor, but not vice versa"
    
    def test_border_detection(self):
        """Test that border cells are properly detected."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test_seed")
        
        # Should have some border cells
        border_count = np.sum(graph.cell_border_flags)
        assert border_count > 0
        assert border_count < len(graph.points)  # Not all cells should be border
    
    def test_vertex_count(self):
        """Test that vertices are generated."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test_seed")
        
        assert len(graph.vertex_coordinates) > 0
        assert len(graph.vertex_neighbors) == len(graph.vertex_coordinates)
        assert len(graph.vertex_cells) == len(graph.vertex_coordinates)
    
    def test_reproducibility(self):
        """Test that same config and seed produce identical results."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        
        graph1 = generate_voronoi_graph(config, "test_seed")
        graph2 = generate_voronoi_graph(config, "test_seed")
        
        np.testing.assert_array_equal(graph1.points, graph2.points)
        assert graph1.cell_neighbors == graph2.cell_neighbors
        np.testing.assert_array_equal(graph1.cell_border_flags, graph2.cell_border_flags)


class TestGridCellFinder:
    """Test grid cell finding function."""
    
    def test_find_valid_cells(self):
        """Test finding cells for coordinates within bounds."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test_seed")
        
        # Test center point
        cell_idx = find_grid_cell(25, 25, graph)
        assert 0 <= cell_idx < len(graph.points)
        
        # Test corner points
        cell_idx = find_grid_cell(5, 5, graph)
        assert 0 <= cell_idx < len(graph.points)
    
    def test_out_of_bounds_clamping(self):
        """Test that out-of-bounds coordinates are clamped."""
        config = GridConfig(width=50, height=50, cells_desired=25)
        graph = generate_voronoi_graph(config, "test_seed")
        
        # Test coordinates outside bounds
        cell_idx = find_grid_cell(-10, -10, graph)
        assert cell_idx == 0
        
        cell_idx = find_grid_cell(1000, 1000, graph)
        assert cell_idx == len(graph.points) - 1


@pytest.mark.parametrize("width,height,cells", [
    (100, 100, 100),
    (200, 150, 500),
    (50, 50, 25)
])
def test_various_grid_sizes(width, height, cells):
    """Test graph generation with various sizes."""
    config = GridConfig(width=width, height=height, cells_desired=cells)
    graph = generate_voronoi_graph(config, "test_seed")
    
    assert len(graph.points) > 0
    assert graph.cells_desired == cells
    assert len(graph.cell_neighbors) == len(graph.points)
    
    # Test that spacing is reasonable
    expected_spacing = np.sqrt((width * height) / cells)
    assert abs(graph.spacing - expected_spacing) < 1.0