"""
DEPRECATED: This is an older partial implementation of reGraph.

The complete, working implementation is now in cell_packing.py.
This file is kept for reference only.

For the current implementation, use:
    from py_fmg.core import regraph  # imports from cell_packing.py
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .voronoi_graph import generate_voronoi_graph, GridConfig


@dataclass
class ReGraphResult:
    """Result of the reGraph operation."""
    points: List[Tuple[float, float]]
    heights: np.ndarray
    grid_indices: np.ndarray
    voronoi_graph: Any


def regraph(voronoi_graph, heights: np.ndarray, config: GridConfig, seed: str) -> ReGraphResult:
    """
    Perform FMG's reGraph operation for coastal resampling.
    
    This function:
    1. Filters out deep ocean cells (height < 20)
    2. Adds intermediate points between coastal neighbors
    3. Returns data for a second Voronoi pass
    
    Args:
        voronoi_graph: Original Voronoi graph with cells and connectivity
        heights: Height values for each cell (after heightmap generation)
        config: Grid configuration
        seed: Random seed for second Voronoi pass
        
    Returns:
        ReGraphResult with new points, heights, and grid indices
    """
    new_points = []
    new_heights = []
    new_grid_indices = []
    
    spacing_squared = voronoi_graph.spacing ** 2
    
    # Process each grid cell (not point!)
    # Grid cells are the original 10,000 cells from Voronoi
    for i in range(len(voronoi_graph.points)):
        height = heights[i]
        
        # Get cell type (1 = land coast, -1 = water coast, 0 = inland, -2 = lake)
        # For now, we'll determine type based on height and neighbors
        cell_type = _get_cell_type(i, heights, voronoi_graph)
        
        # CRITICAL: Exclude ALL deep ocean cells except coastline
        # This is what reduces 10,000 cells to ~4,000
        if height < 20 and cell_type not in [-1, -2]:
            continue
            
        # Exclude non-coastal lake points (type -2)
        # In FMG: type === -2 && (i % 4 === 0 || features[gridCells.f[i]].type === "lake")
        # Since we don't have features yet, we'll skip some lake cells
        if cell_type == -2 and i % 4 == 0:
            continue
        
        # Add the main point
        x, y = voronoi_graph.points[i]
        new_points.append((x, y))
        new_heights.append(height)
        new_grid_indices.append(i)
        
        # Add additional points for cells along coast
        if cell_type in [1, -1]:
            # Check if near border (simplified - skip edge cells)
            if _is_border_cell(i, voronoi_graph):
                continue
                
            # Add intermediate points between same-type neighbors
            for neighbor in voronoi_graph.cell_neighbors[i]:
                if i > neighbor:  # Avoid duplicates
                    continue
                    
                neighbor_type = _get_cell_type(neighbor, heights, voronoi_graph)
                if neighbor_type == cell_type:
                    # Check distance
                    nx, ny = voronoi_graph.points[neighbor]
                    dist_squared = (y - ny) ** 2 + (x - nx) ** 2
                    
                    if dist_squared < spacing_squared:
                        continue  # Too close
                    
                    # Add intermediate point
                    x1 = round((x + nx) / 2, 1)
                    y1 = round((y + ny) / 2, 1)
                    new_points.append((x1, y1))
                    new_heights.append(height)  # Use the same height
                    new_grid_indices.append(i)  # Reference original cell
    
    # Convert to numpy arrays
    new_heights = np.array(new_heights, dtype=np.uint8)
    new_grid_indices = np.array(new_grid_indices, dtype=np.uint32)
    
    # For now, return the data without a second Voronoi pass
    # The packed graph would normally be created here
    # TODO: Implement second Voronoi pass with custom points
    
    # Create a simple packed structure
    from types import SimpleNamespace
    packed_graph = SimpleNamespace(
        points=new_points,
        cell_neighbors=[[] for _ in new_points],  # Empty for now
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        spacing=voronoi_graph.spacing
    )
    
    return ReGraphResult(
        points=new_points,
        heights=new_heights,
        grid_indices=new_grid_indices,
        voronoi_graph=packed_graph
    )


def _get_cell_type(cell_id: int, heights: np.ndarray, graph) -> int:
    """
    Determine cell type based on height and neighbors.
    
    Returns:
        1: Land coast (land cell with water neighbor)
        -1: Water coast (water cell with land neighbor)
        0: Inland (no water neighbors) 
        -2: Lake (simplified - just deep water for now)
    """
    cell_height = heights[cell_id]
    is_water = cell_height < 20
    
    # Check neighbors
    has_water_neighbor = False
    has_land_neighbor = False
    
    for neighbor in graph.cell_neighbors[cell_id]:
        neighbor_height = heights[neighbor]
        if neighbor_height < 20:
            has_water_neighbor = True
        else:
            has_land_neighbor = True
    
    # Determine type - be strict about what counts as coast
    if is_water:
        # Only mark as coast if directly adjacent to land
        if has_land_neighbor and has_water_neighbor:
            return -1  # Water coast
        return -3  # Regular deep ocean - will be excluded!
    else:
        # Land cells are always included
        if has_water_neighbor:
            return 1  # Land coast
        return 0  # Inland


def _is_border_cell(cell_id: int, graph) -> bool:
    """Check if cell is on the map border."""
    x, y = graph.points[cell_id]
    margin = graph.spacing / 2
    
    # Get dimensions from the config that was used to create the graph
    # For now, we'll use a simple heuristic based on max coordinates
    max_x = max(p[0] for p in graph.points)
    max_y = max(p[1] for p in graph.points)
    
    return (x < margin or x > max_x - margin or 
            y < margin or y > max_y - margin)