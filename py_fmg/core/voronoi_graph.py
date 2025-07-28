"""Voronoi graph generation for FMG Python port."""

import numpy as np
from scipy.spatial import Voronoi
from typing import Dict, List, Tuple, NamedTuple
import structlog
from .alea_prng import AleaPRNG

logger = structlog.get_logger()


class GridConfig(NamedTuple):
    """Configuration for grid generation."""
    width: float
    height: float
    cells_desired: int


class VoronoiGraph(NamedTuple):
    """Voronoi graph data structure matching FMG format."""
    spacing: float
    cells_desired: int
    boundary_points: np.ndarray
    points: np.ndarray
    cells_x: int
    cells_y: int
    # Cell connectivity data
    cell_neighbors: List[List[int]]  # cells.c[i] = list of neighbor cell IDs
    cell_vertices: List[List[int]]   # cells.v[i] = list of vertex IDs for cell boundary
    cell_border_flags: np.ndarray    # cells.b[i] = 1 if border cell, 0 otherwise
    # Vertex data
    vertex_coordinates: np.ndarray   # vertices.p[i] = [x, y] coordinates
    vertex_neighbors: List[List[int]] # vertices.v[i] = list of adjacent vertex IDs
    vertex_cells: List[List[int]]    # vertices.c[i] = list of adjacent cell IDs
    seed: str


def get_jittered_grid(width: float, height: float, spacing: float, seed: str = None) -> np.ndarray:
    """
    Generate jittered square grid points.
    
    Equivalent to FMG's getJitteredGrid() function.
    Creates regular grid with randomized positions to prevent artificial patterns.
    
    Args:
        width: Grid width
        height: Grid height  
        spacing: Distance between grid points
        seed: Random seed for reproducibility
        
    Returns:
        Array of [x, y] point coordinates
    """
    # Use Alea PRNG to match FMG exactly
    prng = AleaPRNG(seed) if seed else AleaPRNG("default")
    
    radius = spacing / 2  # square radius
    jittering = radius * 0.9  # max deviation
    double_jittering = jittering * 2
    
    def jitter():
        return prng.random() * double_jittering - jittering
    
    points = []
    y = radius
    while y < height:
        x = radius
        while x < width:
            # Add random jitter matching FMG's implementation
            xj = min(round(x + jitter(), 2), width)
            yj = min(round(y + jitter(), 2), height)
            points.append([xj, yj])
            x += spacing
        y += spacing
    
    return np.array(points)


def get_boundary_points(width: float, height: float, spacing: float) -> np.ndarray:
    """
    Generate boundary points for pseudo-clipping Voronoi cells.
    
    Equivalent to FMG's getBoundaryPoints() function.
    Adds points around map edge to prevent infinite Voronoi cells.
    
    Args:
        width: Grid width
        height: Grid height
        spacing: Base spacing for points
        
    Returns:
        Array of boundary point coordinates
    """
    offset = round(-1 * spacing)  # FMG uses rn(-1 * spacing)
    b_spacing = spacing * 2
    w = width - offset * 2
    h = height - offset * 2
    
    number_x = int(np.ceil(w / b_spacing) - 1)
    number_y = int(np.ceil(h / b_spacing) - 1)
    
    points = []
    
    # Match FMG's loop: for (let i = 0.5; i < numberX; i++)
    for i in range(number_x):
        x = int(np.ceil((w * (i + 0.5)) / number_x + offset))
        points.append([x, offset])
        points.append([x, h + offset])
    
    for i in range(number_y):
        y = int(np.ceil((h * (i + 0.5)) / number_y + offset))
        points.append([offset, y])
        points.append([w + offset, y])
    
    return np.array(points)


def build_cell_connectivity(vor: Voronoi, n_grid_points: int) -> Tuple[List[List[int]], np.ndarray]:
    """
    Build FMG-compatible cell connectivity from scipy Voronoi output.
    
    This is the critical adapter function that converts scipy.spatial.Voronoi
    into the cell-centric graph structure that FMG algorithms expect.
    
    Args:
        vor: scipy Voronoi diagram
        n_grid_points: Number of grid points (excluding boundary)
        
    Returns:
        Tuple of (cell_neighbors, border_flags)
    """
    logger.info(f"Starting build_cell_connectivity with {n_grid_points} grid points")
    cell_neighbors = [[] for _ in range(n_grid_points)]
    border_flags = np.zeros(n_grid_points, dtype=np.uint8)
    
    # Build neighbor relationships from ridge_points
    logger.info(f"Processing {len(vor.ridge_points)} ridge points")
    for ridge_points in vor.ridge_points:
        p1, p2 = ridge_points
        
        # Only process connections between grid points (not boundary points)
        if p1 < n_grid_points and p2 < n_grid_points:
            cell_neighbors[p1].append(p2)
            cell_neighbors[p2].append(p1)
    
    logger.info("Starting border detection")
    
    # First, build a set of cells that connect to boundary points
    border_cells = set()
    for ridge_points in vor.ridge_points:
        p1, p2 = ridge_points
        if p1 < n_grid_points and p2 >= n_grid_points:
            border_cells.add(p1)
        elif p2 < n_grid_points and p1 >= n_grid_points:
            border_cells.add(p2)
    
    # Now process all cells
    for i in range(n_grid_points):
        # Remove duplicates and sort for consistency
        cell_neighbors[i] = sorted(list(set(cell_neighbors[i])))
        
        # Set border flag if this cell connects to boundary
        if i in border_cells:
            border_flags[i] = 1
    
    logger.info("build_cell_connectivity completed")
    return cell_neighbors, border_flags


def build_cell_vertices(vor: Voronoi, n_grid_points: int) -> List[List[int]]:
    """
    Build cell vertex lists from Voronoi diagram.
    
    Each cell gets an ordered list of vertices that form its boundary.
    
    Args:
        vor: scipy Voronoi diagram
        n_grid_points: Number of grid points
        
    Returns:
        List of vertex ID lists for each cell
    """
    cell_vertices = [[] for _ in range(n_grid_points)]
    
    # Map each cell to its bounding vertices
    for ridge_idx, ridge_vertices in enumerate(vor.ridge_vertices):
        if -1 in ridge_vertices:  # Skip infinite ridges
            continue
            
        ridge_points = vor.ridge_points[ridge_idx]
        p1, p2 = ridge_points
        
        # Add vertices to both cells that share this ridge
        if p1 < n_grid_points:
            cell_vertices[p1].extend(ridge_vertices)
        if p2 < n_grid_points:
            cell_vertices[p2].extend(ridge_vertices)
    
    # Remove duplicates and ensure proper ordering
    for i in range(n_grid_points):
        if cell_vertices[i]:
            # Remove duplicates while preserving order
            seen = set()
            ordered_vertices = []
            for v in cell_vertices[i]:
                if v not in seen:
                    seen.add(v)
                    ordered_vertices.append(v)
            cell_vertices[i] = ordered_vertices
    
    return cell_vertices


def build_vertex_connectivity(vor: Voronoi, n_grid_points: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build vertex connectivity from Voronoi diagram.
    
    Args:
        vor: scipy Voronoi diagram
        n_grid_points: Number of grid points (excluding boundary)
        
    Returns:
        Tuple of (vertex_neighbors, vertex_cells)
    """
    n_vertices = len(vor.vertices)
    vertex_neighbors = [[] for _ in range(n_vertices)]
    vertex_cells = [[] for _ in range(n_vertices)]
    
    # Build vertex-vertex connections from ridges
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:  # Skip infinite ridges
            continue
        if len(ridge_vertices) == 2:
            v1, v2 = ridge_vertices
            vertex_neighbors[v1].append(v2)
            vertex_neighbors[v2].append(v1)
    
    # Build vertex-cell connections (only for grid cells, not boundary)
    for cell_idx, cell_vertices in enumerate(build_cell_vertices(vor, n_grid_points)):
        for vertex_idx in cell_vertices:
            if vertex_idx < n_vertices:
                vertex_cells[vertex_idx].append(cell_idx)
    
    # Clean up duplicates
    for i in range(n_vertices):
        vertex_neighbors[i] = sorted(list(set(vertex_neighbors[i])))
        vertex_cells[i] = sorted(list(set(vertex_cells[i])))
    
    return vertex_neighbors, vertex_cells


def generate_voronoi_graph(config: GridConfig, seed: str = None) -> VoronoiGraph:
    """
    Generate complete Voronoi graph matching FMG structure.
    
    This is the main function that replaces FMG's generateGrid() function.
    
    Args:
        config: Grid configuration
        seed: Random seed for reproducibility
        
    Returns:
        Complete Voronoi graph data structure
    """
    logger.info("Generating Voronoi graph", 
                width=config.width, height=config.height, 
                cells_desired=config.cells_desired, seed=seed)
    
    # Calculate grid parameters
    spacing = np.sqrt((config.width * config.height) / config.cells_desired)
    spacing = round(spacing, 2)
    
    cells_x = int((config.width + 0.5 * spacing - 1e-10) / spacing)
    cells_y = int((config.height + 0.5 * spacing - 1e-10) / spacing)
    
    logger.info("Grid parameters calculated", 
                spacing=spacing, cells_x=cells_x, cells_y=cells_y)
    
    # Generate points
    grid_points = get_jittered_grid(config.width, config.height, spacing, seed)
    boundary_points = get_boundary_points(config.width, config.height, spacing)
    all_points = np.vstack([grid_points, boundary_points])
    
    logger.info("Points generated", 
                grid_points=len(grid_points), 
                boundary_points=len(boundary_points))
    
    # Calculate Voronoi diagram
    vor = Voronoi(all_points)
    
    logger.info("Voronoi diagram calculated", 
                vertices=len(vor.vertices), ridges=len(vor.ridge_points))
    
    # Build FMG-compatible data structures
    logger.info("Building cell connectivity...")
    cell_neighbors, border_flags = build_cell_connectivity(vor, len(grid_points))
    logger.info("Building cell vertices...")
    cell_vertices = build_cell_vertices(vor, len(grid_points))
    logger.info("Building vertex connectivity...")
    vertex_neighbors, vertex_cells = build_vertex_connectivity(vor, len(grid_points))
    logger.info("All connectivity built")
    
    return VoronoiGraph(
        spacing=spacing,
        cells_desired=config.cells_desired,
        boundary_points=boundary_points,
        points=grid_points,
        cells_x=cells_x,
        cells_y=cells_y,
        cell_neighbors=cell_neighbors,
        cell_vertices=cell_vertices,
        cell_border_flags=border_flags,
        vertex_coordinates=vor.vertices,
        vertex_neighbors=vertex_neighbors,
        vertex_cells=vertex_cells,
        seed=seed or "default"
    )


def find_grid_cell(x: float, y: float, graph: VoronoiGraph) -> int:
    """
    Find the cell index for given coordinates.
    
    Equivalent to FMG's findGridCell() function.
    Uses approximate grid-based lookup for performance.
    
    Args:
        x, y: Coordinates to find
        graph: Voronoi graph
        
    Returns:
        Cell index containing the point
    """
    # Approximate grid-based lookup
    grid_x = int(x / graph.spacing)
    grid_y = int(y / graph.spacing)
    
    # Calculate approximate cell index
    cell_idx = grid_y * graph.cells_x + grid_x
    
    # Clamp to valid range
    return min(max(cell_idx, 0), len(graph.points) - 1)


def pack_graph(graph: VoronoiGraph, heights: np.ndarray, deep_ocean_threshold: int = 20) -> VoronoiGraph:
    """
    Pack the Voronoi graph by removing deep ocean cells.
    
    This function replicates FMG's reGraph() functionality which optimizes the graph
    by removing deep ocean cells (typically height < 20) to reduce memory usage
    and improve performance for subsequent processing stages.
    
    Args:
        graph: Original Voronoi graph
        heights: Height values for each cell
        deep_ocean_threshold: Cells with height below this value are removed
        
    Returns:
        New packed VoronoiGraph with reduced cell count
    """
    logger.info("Packing graph", 
                original_cells=len(graph.points), 
                threshold=deep_ocean_threshold)
    
    # Find cells to keep (above threshold)
    keep_mask = heights >= deep_ocean_threshold
    keep_indices = np.where(keep_mask)[0]
    n_packed = len(keep_indices)
    
    logger.info("Cells to pack", kept=n_packed, removed=len(graph.points) - n_packed)
    
    # Create old_to_new index mapping
    old_to_new = np.full(len(graph.points), -1, dtype=int)
    old_to_new[keep_indices] = np.arange(n_packed)
    
    # Pack points array
    packed_points = graph.points[keep_indices]
    
    # Pack cell connectivity - only keep neighbors that are also kept
    packed_neighbors = []
    for old_idx in keep_indices:
        old_neighbors = graph.cell_neighbors[old_idx]
        new_neighbors = []
        for neighbor_idx in old_neighbors:
            if keep_mask[neighbor_idx]:  # Neighbor is kept
                new_neighbors.append(old_to_new[neighbor_idx])
        packed_neighbors.append(new_neighbors)
    
    # Pack cell vertices - need to rebuild vertex system
    # For simplicity, we'll keep all vertices but update cell references
    packed_vertices = []
    for old_idx in keep_indices:
        packed_vertices.append(graph.cell_vertices[old_idx])
    
    # Pack border flags
    packed_border_flags = graph.cell_border_flags[keep_indices]
    
    # Update vertex connectivity to reference packed cells
    packed_vertex_neighbors = []
    packed_vertex_cells = []
    
    for v_idx in range(len(graph.vertex_neighbors)):
        # Keep vertex neighbors as-is (vertex indices don't change)
        packed_vertex_neighbors.append(graph.vertex_neighbors[v_idx])
        
        # Update vertex cells to only reference packed cells
        old_cells = graph.vertex_cells[v_idx]
        new_cells = []
        for cell_idx in old_cells:
            if keep_mask[cell_idx]:
                new_cells.append(old_to_new[cell_idx])
        packed_vertex_cells.append(new_cells)
    
    # Calculate new approximate grid dimensions for the packed cells
    packed_width = np.max(packed_points[:, 0]) - np.min(packed_points[:, 0])
    packed_height = np.max(packed_points[:, 1]) - np.min(packed_points[:, 1])
    packed_cells_x = int(packed_width / graph.spacing) + 1
    packed_cells_y = int(packed_height / graph.spacing) + 1
    
    logger.info("Graph packing complete", 
                packed_cells=n_packed,
                original_cells=len(graph.points),
                reduction_pct=round((1 - n_packed/len(graph.points)) * 100, 1))
    
    return VoronoiGraph(
        spacing=graph.spacing,
        cells_desired=n_packed,  # Update to reflect new reality
        boundary_points=graph.boundary_points,  # Keep boundary unchanged
        points=packed_points,
        cells_x=packed_cells_x,
        cells_y=packed_cells_y,
        cell_neighbors=packed_neighbors,
        cell_vertices=packed_vertices,
        cell_border_flags=packed_border_flags,
        vertex_coordinates=graph.vertex_coordinates,  # Keep all vertices
        vertex_neighbors=packed_vertex_neighbors,
        vertex_cells=packed_vertex_cells,
        seed=graph.seed + "_packed"  # Indicate this is packed version
    )