"""Voronoi graph generation for FMG Python port."""

import numpy as np
from scipy.spatial import Voronoi
from typing import Dict, List, Tuple, NamedTuple, Optional
from dataclasses import dataclass, field
import structlog
from .alea_prng import AleaPRNG

logger = structlog.get_logger()


class GridConfig(NamedTuple):
    """Configuration for grid generation."""
    width: float
    height: float
    cells_desired: int


@dataclass
class VoronoiGraph:
    """Voronoi graph data structure matching FMG format.
    
    This is now a mutable dataclass to support FMG's stateful operations,
    including height pre-allocation and grid reuse.
    """
    # Grid parameters
    spacing: float
    cells_desired: int
    graph_width: float  # Original generation width
    graph_height: float  # Original generation height
    seed: str
    
    # Points data
    boundary_points: np.ndarray
    points: np.ndarray
    cells_x: int
    cells_y: int
    
    # Cell connectivity data
    cell_neighbors: List[List[int]]  # cells.c[i] = list of neighbor cell IDs
    cell_vertices: List[List[int]]   # cells.v[i] = list of vertex IDs for cell boundary
    cell_border_flags: np.ndarray    # cells.b[i] = 1 if border cell, 0 otherwise
    heights: np.ndarray              # cells.h[i] = height value (PRE-ALLOCATED!)
    
    # Vertex data
    vertex_coordinates: np.ndarray   # vertices.p[i] = [x, y] coordinates
    vertex_neighbors: List[List[int]] # vertices.v[i] = list of adjacent vertex IDs
    vertex_cells: List[List[int]]    # vertices.c[i] = list of adjacent cell IDs
    
    # Optional: Mapping from packed cells to original grid (set by regraph)
    grid_indices: Optional[np.ndarray] = field(default=None)
    
    # Feature fields (populated by features module)
    distance_field: Optional[np.ndarray] = field(default=None)  # cells.t - distance to coast
    feature_ids: Optional[np.ndarray] = field(default=None)     # cells.f - feature ID for each cell
    features: Optional[List] = field(default=None)              # array of feature objects
    border_cells: Optional[np.ndarray] = field(default=None)    # cells.b - border cell flags
    
    def should_regenerate(self, config: GridConfig, seed: str) -> bool:
        """Check if grid needs to be regenerated based on new parameters.
        
        Equivalent to FMG's shouldRegenerateGrid() function.
        """
        same_size = (self.graph_width == config.width and 
                    self.graph_height == config.height)
        same_cells = self.cells_desired == config.cells_desired
        same_seed = self.seed == seed
        
        return not (same_size and same_cells and same_seed)


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


def compute_polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Compute the centroid of a polygon.
    
    Args:
        vertices: Array of [x, y] vertex coordinates
        
    Returns:
        [x, y] centroid coordinates
    """
    if len(vertices) < 3:
        return np.mean(vertices, axis=0)
    
    # Calculate area and centroid using shoelace formula
    n = len(vertices)
    area = 0.0
    cx = 0.0
    cy = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        a = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
        area += a
        cx += (vertices[i][0] + vertices[j][0]) * a
        cy += (vertices[i][1] + vertices[j][1]) * a
    
    if abs(area) < 1e-10:
        return np.mean(vertices, axis=0)
    
    area *= 0.5
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    
    return np.array([cx, cy])


def relax_points(points: np.ndarray, boundary_points: np.ndarray, 
                 width: float, height: float, n_iterations: int = 3) -> np.ndarray:
    """Apply Lloyd's relaxation to improve point distribution.
    
    Equivalent to FMG's relaxPlaced() function.
    Moves each point to the centroid of its Voronoi cell.
    
    Args:
        points: Grid points to relax
        boundary_points: Boundary points (fixed)
        width: Map width
        height: Map height  
        n_iterations: Number of relaxation iterations
        
    Returns:
        Relaxed point coordinates
    """
    logger.info("Starting Lloyd's relaxation", iterations=n_iterations)
    
    points = points.copy()  # Don't modify original
    n_points = len(points)
    
    for iteration in range(n_iterations):
        all_points = np.vstack([points, boundary_points])
        vor = Voronoi(all_points)
        
        # Move each point to its cell's centroid
        for i in range(n_points):
            # Get vertices of this cell's polygon
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
                
            region_vertices = vor.regions[region_idx]
            if -1 in region_vertices or len(region_vertices) < 3:
                continue
            
            # Get vertex coordinates
            vertices = vor.vertices[region_vertices]
            
            # Compute centroid and update point
            centroid = compute_polygon_centroid(vertices)
            
            # Clamp to map bounds
            points[i][0] = np.clip(centroid[0], 0, width)
            points[i][1] = np.clip(centroid[1], 0, height)
        
        logger.info(f"Relaxation iteration {iteration + 1} complete")
    
    return points


def generate_or_reuse_grid(existing_grid: Optional[VoronoiGraph], 
                          config: GridConfig, seed: str = None,
                          apply_relaxation: bool = True) -> VoronoiGraph:
    """
    Generate new grid or reuse existing one based on parameters.
    
    Equivalent to FMG's logic in applyGraphSize() and generateGrid().
    Allows keeping landmasses while changing terrain features.
    
    Args:
        existing_grid: Previously generated grid (can be None)
        config: Grid configuration
        seed: Random seed
        apply_relaxation: Whether to apply Lloyd's relaxation
        
    Returns:
        VoronoiGraph - either new or reused
    """
    # Check if we should regenerate
    if existing_grid is None or existing_grid.should_regenerate(config, seed):
        logger.info("Generating new grid")
        return generate_voronoi_graph(config, seed, apply_relaxation)
    else:
        logger.info("Reusing existing grid", 
                   seed=existing_grid.seed,
                   cells=len(existing_grid.points))
        return existing_grid


def generate_voronoi_graph(config: GridConfig, seed: str = None, 
                          apply_relaxation: bool = True) -> VoronoiGraph:
    """
    Generate complete Voronoi graph matching FMG structure.
    
    This is the main function that replaces FMG's generateGrid() function.
    Now includes Lloyd's relaxation and height pre-allocation.
    
    Args:
        config: Grid configuration
        seed: Random seed for reproducibility
        apply_relaxation: Whether to apply Lloyd's relaxation
        
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
    
    # Apply Lloyd's relaxation if requested
    if apply_relaxation:
        grid_points = relax_points(grid_points, boundary_points, 
                                  config.width, config.height, n_iterations=3)
    
    all_points = np.vstack([grid_points, boundary_points])
    
    logger.info("Points generated", 
                grid_points=len(grid_points), 
                boundary_points=len(boundary_points),
                relaxation_applied=apply_relaxation)
    
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
    
    # PRE-ALLOCATE HEIGHTS ARRAY - Critical for FMG compatibility!
    heights = np.zeros(len(grid_points), dtype=np.uint8)
    logger.info("Heights array pre-allocated")
    
    return VoronoiGraph(
        spacing=spacing,
        cells_desired=config.cells_desired,
        graph_width=config.width,
        graph_height=config.height,
        seed=seed or "default",
        boundary_points=boundary_points,
        points=grid_points,
        cells_x=cells_x,
        cells_y=cells_y,
        cell_neighbors=cell_neighbors,
        cell_vertices=cell_vertices,
        cell_border_flags=border_flags,
        heights=heights,
        vertex_coordinates=vor.vertices,
        vertex_neighbors=vertex_neighbors,
        vertex_cells=vertex_cells
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
