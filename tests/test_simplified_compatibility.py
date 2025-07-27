"""
Simplified FMG Compatibility Test

A more focused test that verifies our Voronoi topology is structurally equivalent
to FMG, without requiring exact coordinate matching.
"""

import json
import pytest
import numpy as np
from pathlib import Path

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph


class TestSimplifiedCompatibility:
    """Test topology compatibility without requiring exact coordinate matching."""
    
    @classmethod  
    def setup_class(cls):
        """Load FMG reference data."""
        reference_path = Path(__file__).parent / "Mateau Full 2025-07-27-14-53.json"
        
        if not reference_path.exists():
            pytest.skip(f"Reference file not found: {reference_path}")
        
        with open(reference_path, 'r') as f:
            cls.fmg_data = json.load(f)
        
        cls.seed = str(cls.fmg_data['info']['seed'])
        cls.width = cls.fmg_data['info']['width']
        cls.height = cls.fmg_data['info']['height']
        cls.grid = cls.fmg_data['grid']
        cls.cells_desired = cls.grid['cellsDesired']
        
        print(f"Reference: seed={cls.seed}, {cls.width}x{cls.height}, cells_desired={cls.cells_desired}")
    
    def test_basic_parameters_match(self):
        """Test that we can match FMG's basic grid parameters."""
        config = GridConfig(
            width=self.width,
            height=self.height,
            cells_desired=self.cells_desired
        )
        
        python_graph = generate_voronoi_graph(config, seed=self.seed)
        
        # Should generate same number of points
        assert len(python_graph.points) == len(self.grid['points'])
        
        # Grid parameters should match
        assert python_graph.spacing == self.grid['spacing']
        assert python_graph.cells_x == self.grid['cellsX']
        assert python_graph.cells_y == self.grid['cellsY']
        
        print(f"✅ Grid parameters match: {python_graph.cells_x}x{python_graph.cells_y}, spacing={python_graph.spacing}")
    
    def test_topology_structure_consistency(self):
        """Test that topology has reasonable structure compared to FMG."""
        config = GridConfig(
            width=self.width,
            height=self.height, 
            cells_desired=self.cells_desired
        )
        
        python_graph = generate_voronoi_graph(config, seed=self.seed)
        fmg_grid_cells = self.grid['cells']
        
        # Test neighbor count distribution
        py_neighbor_counts = [len(neighbors) for neighbors in python_graph.cell_neighbors]
        fmg_neighbor_counts = [len(cell['c']) for cell in fmg_grid_cells]
        
        py_avg_neighbors = np.mean(py_neighbor_counts)
        fmg_avg_neighbors = np.mean(fmg_neighbor_counts)
        
        print(f"Average neighbors - Python: {py_avg_neighbors:.2f}, FMG: {fmg_avg_neighbors:.2f}")
        
        # Should be within reasonable range (Voronoi cells typically have 6 neighbors)
        assert 5.0 < py_avg_neighbors < 7.0
        assert abs(py_avg_neighbors - fmg_avg_neighbors) < 0.5
        
        # Test neighbor count distributions are similar
        py_counts = np.bincount(py_neighbor_counts)
        fmg_counts = np.bincount(fmg_neighbor_counts)
        
        # Pad to same length for comparison
        max_len = max(len(py_counts), len(fmg_counts))
        py_counts = np.pad(py_counts, (0, max_len - len(py_counts)))
        fmg_counts = np.pad(fmg_counts, (0, max_len - len(fmg_counts)))
        
        # Normalize to compare distributions
        py_dist = py_counts / np.sum(py_counts)
        fmg_dist = fmg_counts / np.sum(fmg_counts)
        
        # Distributions should be reasonably similar
        kl_divergence = np.sum(py_dist * np.log(py_dist / (fmg_dist + 1e-10) + 1e-10))
        print(f"Neighbor count distribution KL divergence: {kl_divergence:.6f}")
        
        assert kl_divergence < 0.1, f"Topology distributions too different: {kl_divergence}"
        
        print("✅ Topology structure is consistent with FMG")
    
    def test_border_cell_detection_reasonable(self):
        """Test that border cell detection produces reasonable results."""
        config = GridConfig(
            width=self.width,
            height=self.height,
            cells_desired=self.cells_desired
        )
        
        python_graph = generate_voronoi_graph(config, seed=self.seed)
        
        border_count = np.sum(python_graph.cell_border_flags)
        total_cells = len(python_graph.points)
        border_percentage = border_count / total_cells * 100
        
        print(f"Border cells: {border_count}/{total_cells} ({border_percentage:.1f}%)")
        
        # For a 100x100 grid, expect roughly 400 border cells (perimeter)
        expected_border_percentage = 4.0  # Rough estimate for square grid
        
        assert 2.0 < border_percentage < 8.0, f"Border percentage seems wrong: {border_percentage:.1f}%"
        
        print("✅ Border cell detection reasonable")
    
    def test_reproducibility(self):
        """Test that same config produces identical results."""
        config = GridConfig(
            width=self.width,
            height=self.height,
            cells_desired=self.cells_desired
        )
        
        graph1 = generate_voronoi_graph(config, seed=self.seed)
        graph2 = generate_voronoi_graph(config, seed=self.seed)
        
        # Should be identical
        np.testing.assert_array_equal(graph1.points, graph2.points)
        assert graph1.cell_neighbors == graph2.cell_neighbors
        
        print("✅ Reproducibility verified")
    
    def test_points_in_reasonable_range(self):
        """Test that generated points are within expected coordinate ranges."""
        config = GridConfig(
            width=self.width,
            height=self.height,
            cells_desired=self.cells_desired
        )
        
        python_graph = generate_voronoi_graph(config, seed=self.seed)
        
        # All points should be within bounds
        assert np.all(python_graph.points[:, 0] >= 0)
        assert np.all(python_graph.points[:, 0] <= self.width)
        assert np.all(python_graph.points[:, 1] >= 0)
        assert np.all(python_graph.points[:, 1] <= self.height)
        
        # Points should be reasonably distributed
        x_range = np.ptp(python_graph.points[:, 0])  # Peak-to-peak
        y_range = np.ptp(python_graph.points[:, 1])
        
        assert x_range > self.width * 0.8  # Should span most of width
        assert y_range > self.height * 0.8  # Should span most of height
        
        print(f"✅ Points well distributed: x_range={x_range:.1f}, y_range={y_range:.1f}")


# Quick test to verify file loading works
def test_reference_file_loads():
    """Quick test that reference file can be loaded."""
    reference_path = Path(__file__).parent / "Mateau Full 2025-07-27-14-53.json"
    
    if not reference_path.exists():
        pytest.skip("Reference file not found")
    
    with open(reference_path, 'r') as f:
        data = json.load(f)
    
    assert 'info' in data
    assert 'grid' in data
    assert data['info']['seed'] == "651658815"
    
    print("✅ Reference file loads correctly")