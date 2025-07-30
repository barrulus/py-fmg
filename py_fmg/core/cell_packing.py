"""
Cell packing (reGraph) functionality matching FMG's implementation.

This module implements FMG's reGraph() function which:
1. Filters out deep ocean cells
2. Adds intermediate points along coastlines
3. Creates a new packed Voronoi graph with fewer, more relevant cells

This is a critical performance optimization that reduces ~10,000 cells to ~4,500
by removing deep ocean while enhancing coastal detail.
"""

import numpy as np
from scipy.spatial import Voronoi
from dataclasses import dataclass
import structlog

from .voronoi_graph import VoronoiGraph, build_cell_connectivity, build_cell_vertices, build_vertex_connectivity

logger = structlog.get_logger()


@dataclass
class CellType:
    """Cell type classification for reGraph."""
    INLAND = 0          # Land cell with no water neighbors
    LAND_COAST = 1      # Land cell with water neighbors
    WATER_COAST = -1    # Water cell with land neighbors  
    LAKE = -2           # Lake cell (simplified)
    DEEP_OCEAN = -3     # Deep ocean (to be excluded)


def regraph(graph: VoronoiGraph) -> VoronoiGraph:
    """
    Perform FMG's reGraph operation to create a packed cell structure.
    
    This function:
    1. Uses cell types from Features.markup_grid (via graph.distance_field)
    2. Filters out deep ocean cells
    3. Adds intermediate points along coastlines
    4. Creates a new Voronoi diagram from the filtered points
    
    This reduces the cell count from ~10,000 to ~4,500 while improving
    coastal detail - a critical optimization for browser performance.
    
    Args:
        graph: Original VoronoiGraph with heights calculated and 
               distance_field populated by Features.markup_grid()
        
    Returns:
        New packed VoronoiGraph with fewer cells and enhanced coastlines
    """
    logger.info("Starting reGraph operation", original_cells=len(graph.points))
    
    # Use the distance_field populated by Features.markup_grid()
    if graph.distance_field is None:
        raise ValueError("graph.distance_field not found. Call Features.markup_grid() first!")
    
    cell_types = graph.distance_field
    
    # Collect new points
    new_points = []
    new_heights = []
    new_grid_indices = []  # Maps packed cells back to original grid
    
    spacing_squared = graph.spacing ** 2
    
    # Process each cell
    for i in range(len(graph.points)):
        cell_type = cell_types[i]
        height = graph.heights[i]
        
        # CRITICAL: Match FMG's exact exclusion logic
        # if (height < 20 && type !== -1 && type !== -2) continue;
        if height < 20 and cell_type != -1 and cell_type != -2:
            continue  # exclude all deep ocean points
            
        # if (type === -2 && (i % 4 === 0 || features[gridCells.f[i]].type === "lake")) continue;
        # For now, we'll just check i % 4 since we don't have lake feature detection yet
        if cell_type == -2 and i % 4 == 0:
            continue  # exclude non-coastal lake points
        
        # Add the main point
        x, y = graph.points[i]
        new_points.append([x, y])
        new_heights.append(height)
        new_grid_indices.append(i)
        
        # Add additional points for cells along coast
        # LAND_COAST = 1, WATER_COAST = -1
        if cell_type == 1 or cell_type == -1:
            # Skip border cells to avoid edge artifacts
            if graph.cell_border_flags[i]:
                continue
                
            # Add intermediate points between same-type coastal neighbors
            # FMG uses forEach on neighbors, checking each one
            for j, neighbor_idx in enumerate(graph.cell_neighbors[i]):
                # FMG: if (i > e) return; - only process once per pair
                if i > neighbor_idx:
                    continue
                    
                neighbor_type = cell_types[neighbor_idx]
                
                # FMG: if (gridCells.t[e] === type)
                if neighbor_type == cell_type:
                    nx, ny = graph.points[neighbor_idx]
                    
                    # Check distance - FMG: const dist2 = (y - points[e][1]) ** 2 + (x - points[e][0]) ** 2;
                    dist_squared = (y - ny) ** 2 + (x - nx) ** 2
                    if dist_squared < spacing_squared:
                        continue  # Too close
                    
                    # Add intermediate point - FMG uses rn() for rounding
                    x1 = round((x + nx) / 2, 1)
                    y1 = round((y + ny) / 2, 1)
                    
                    new_points.append([x1, y1])
                    new_heights.append(height)  # Use first cell's height
                    new_grid_indices.append(i)  # Reference first cell
    
    logger.info("Points collected for packing", 
                original=len(graph.points), 
                packed=len(new_points),
                reduction_pct=round((1 - len(new_points)/len(graph.points)) * 100, 1))
    
    # Convert to numpy arrays
    new_points = np.array(new_points)
    new_heights = np.array(new_heights, dtype=np.uint8)
    new_grid_indices = np.array(new_grid_indices, dtype=np.uint32)
    
    # Create new Voronoi diagram from packed points
    # Combine with boundary points for proper edge handling
    all_points = np.vstack([new_points, graph.boundary_points])
    vor = Voronoi(all_points)
    
    logger.info("New Voronoi diagram created", 
                vertices=len(vor.vertices), 
                ridges=len(vor.ridge_points))
    
    # Build connectivity for packed cells
    n_packed = len(new_points)
    cell_neighbors, border_flags = build_cell_connectivity(vor, n_packed)
    cell_vertices = build_cell_vertices(vor, n_packed)
    vertex_neighbors, vertex_cells = build_vertex_connectivity(vor, n_packed)
    
    # Calculate new grid dimensions
    width = np.max(new_points[:, 0]) - np.min(new_points[:, 0])
    height_dim = np.max(new_points[:, 1]) - np.min(new_points[:, 1])
    cells_x = int(width / graph.spacing) + 1
    cells_y = int(height_dim / graph.spacing) + 1
    
    # Create packed VoronoiGraph
    packed = VoronoiGraph(
        spacing=graph.spacing,
        cells_desired=n_packed,
        graph_width=graph.graph_width,  # Keep original dimensions
        graph_height=graph.graph_height,
        seed=graph.seed + "_packed",
        boundary_points=graph.boundary_points,
        points=new_points,
        cells_x=cells_x,
        cells_y=cells_y,
        cell_neighbors=cell_neighbors,
        cell_vertices=cell_vertices,
        cell_border_flags=border_flags,
        heights=new_heights,
        vertex_coordinates=vor.vertices,
        vertex_neighbors=vertex_neighbors,
        vertex_cells=vertex_cells
    )
    
    # Store the mapping from packed cells to original grid cells
    # This is critical for transferring data between grids
    packed.grid_indices = new_grid_indices
    
    logger.info("reGraph complete", 
                packed_cells=n_packed,
                coastal_points_added=n_packed - np.sum(new_grid_indices == np.arange(len(new_grid_indices))))
    
    return packed

