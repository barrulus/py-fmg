"""Voronoi graph generation for FMG Python port."""

import numpy as np
from scipy.spatial import Voronoi
from typing import Dict, List, Tuple, NamedTuple
import structlog

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
    if seed:
        np.random.seed(hash(seed) % (2**32))
    
    radius = spacing / 2  # Grid cell radius
    jittering = radius * 0.9  # Maximum deviation (45% of spacing)
    
    points = []
    y = radius
    while y < height:
        x = radius
        while x < width:
            # Add random jitter
            x_jittered = min(x + np.random.uniform(-jittering, jittering), width)
            y_jittered = min(y + np.random.uniform(-jittering, jittering), height)
            points.append([x_jittered, y_jittered])
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
    offset = -spacing  # Move boundary outward
    b_spacing = spacing * 2  # Larger spacing for boundary
    w = width - offset * 2
    h = height - offset * 2
    
    number_x = max(1, int(np.ceil(w / b_spacing)) - 1)
    number_y = max(1, int(np.ceil(h / b_spacing)) - 1)
    
    points = []
    
    # Top and bottom edges
    for i in range(number_x):
        x = int((w * (i + 0.5)) / number_x + offset)
        points.extend([[x, offset], [x, h + offset]])
    
    # Left and right edges  
    for i in range(number_y):
        y = int((h * (i + 0.5)) / number_y + offset)
        points.extend([[offset, y], [w + offset, y]])
    
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
    cell_neighbors = [[] for _ in range(n_grid_points)]
    border_flags = np.zeros(n_grid_points, dtype=np.uint8)
    
    # Build neighbor relationships from ridge_points
    for ridge_points in vor.ridge_points:
        p1, p2 = ridge_points
        
        # Only process connections between grid points (not boundary points)
        if p1 < n_grid_points and p2 < n_grid_points:
            cell_neighbors[p1].append(p2)
            cell_neighbors[p2].append(p1)
    
    # Detect border cells (fewer neighbors than expected)
    for i in range(n_grid_points):
        # Remove duplicates and sort for consistency
        cell_neighbors[i] = sorted(list(set(cell_neighbors[i])))
        
        # Border detection: cells with connections to boundary points
        for ridge_points in vor.ridge_points:
            p1, p2 = ridge_points
            if (p1 == i and p2 >= n_grid_points) or (p2 == i and p1 >= n_grid_points):
                border_flags[i] = 1
                break
    
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


def build_vertex_connectivity(vor: Voronoi) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build vertex connectivity from Voronoi diagram.
    
    Args:
        vor: scipy Voronoi diagram
        
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
    
    # Build vertex-cell connections
    for cell_idx, cell_vertices in enumerate(build_cell_vertices(vor, len(vor.points))):
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
    cell_neighbors, border_flags = build_cell_connectivity(vor, len(grid_points))
    cell_vertices = build_cell_vertices(vor, len(grid_points))
    vertex_neighbors, vertex_cells = build_vertex_connectivity(vor)
    
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