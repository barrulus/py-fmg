"""
Hydrology system for river generation and water flow simulation.

This module implements:
- Depression filling algorithm
- Water flow simulation and accumulation
- River generation with meandering
- Lake detection in closed basins
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import structlog
from collections import deque
import heapq

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HydrologyOptions:
    """Hydrology calculation options matching FMG's parameters."""
    min_river_flow: float = 30.0  # Minimum flow to form a river
    lake_threshold: float = 0.85  # Threshold for lake formation
    river_width_factor: float = 1.0  # Factor for river width calculation
    meander_factor: float = 0.3  # Factor for river meandering
    evaporation_rate: float = 0.1  # Water evaporation rate


@dataclass
class River:
    """Represents a river with its properties."""
    id: int
    cells: List[int]  # Cell indices forming the river
    flow: float  # Water flow volume
    length: float  # River length
    source_cell: int  # Source cell index
    mouth_cell: int  # Mouth cell index
    tributaries: List[int] = None  # List of tributary river IDs
    
    def __post_init__(self):
        if self.tributaries is None:
            self.tributaries = []


@dataclass
class Lake:
    """Represents a lake with its properties."""
    id: int
    cells: Set[int]  # Cell indices forming the lake
    water_level: float  # Water level height
    outlet_cell: Optional[int] = None  # Outlet cell if any
    area: float = 0.0  # Lake area
    sea_level: int = 20  # Height threshold for water
    min_river_flux: float = 30.0  # Minimum flow to form visible river
    max_depression_iterations: int = 100  # Max iterations for depression resolution
    lake_elevation_increment: float = 0.2  # Height increment for persistent lakes
    depression_elevation_increment: float = 0.1  # Height increment for depressions
    meandering_factor: float = 0.5  # Base factor for river meandering
    width_scale_factor: float = 1.0  # Scale factor for river width calculation

    # Enhanced hydraulic parameters
    manning_n: float = 0.035  # Manning's roughness coefficient (natural channels)
    depth_width_ratio: float = 0.1  # Typical depth/width ratio for rivers
    min_slope: float = 0.0001  # Minimum slope to prevent division by zero


@dataclass
class RiverData:
    """Data structure for a river segment."""
    id: int
    cells: List[int] = field(default_factory=list)
    parent_id: Optional[int] = None
    discharge: float = 0.0
    width: float = 0.0
    length: float = 0.0
    source_distance: float = 0.0



class Hydrology:
    """Handles water flow simulation and river generation."""

    
    def __init__(self, graph, options: Optional[HydrologyOptions] = None):
        """
        Initialize hydrology system.
        
        Args:
            graph: VoronoiGraph with heights and precipitation populated
            options: Hydrology calculation options
        """
        self.graph = graph
        self.options = options or HydrologyOptions()
        
        # Water flow arrays
        self.water_flux = None  # Water accumulation at each cell
        self.flow_directions = None  # Flow direction for each cell
        self.filled_heights = None  # Heights after depression filling
        
        # Generated features
        self.rivers = []  # List of River objects
        self.lakes = []  # List of Lake objects
        
    def fill_depressions(self):
        """
        Fill depressions in the heightmap to ensure proper drainage.
        
        This implements a priority queue-based depression filling algorithm
        similar to FMG's approach.
        """
        logger.info("Filling depressions")
        
        n_cells = len(self.graph.points)
        self.filled_heights = self.graph.heights.astype(np.float32).copy()
        
        # Priority queue: (height, cell_index)
        pq = []
        processed = np.zeros(n_cells, dtype=bool)
        
        # Initialize with border cells (they can drain off the map)
        for i in range(n_cells):
            if self.graph.cell_border_flags[i]:
                heapq.heappush(pq, (self.filled_heights[i], i))
                processed[i] = True
                
        # Process cells in order of increasing height
        while pq:
            current_height, cell_idx = heapq.heappop(pq)
            
            # Process neighbors
            for neighbor_idx in self.graph.cell_neighbors[cell_idx]:
                if not processed[neighbor_idx]:
                    neighbor_height = self.filled_heights[neighbor_idx]
                    
                    # If neighbor is lower than current, it's in a depression
                    # Fill it to the current height
                    if neighbor_height < current_height:
                        self.filled_heights[neighbor_idx] = current_height
                        heapq.heappush(pq, (current_height, neighbor_idx))
                    else:
                        heapq.heappush(pq, (neighbor_height, neighbor_idx))
                    
                    processed[neighbor_idx] = True
                    
        logger.info("Depression filling completed", 
                   cells_filled=np.sum(self.filled_heights > self.graph.heights))
        
    def calculate_flow_directions(self):
        """
        Calculate flow direction for each cell based on filled heights.
        
        Each cell flows to its lowest neighbor.
        """
        logger.info("Calculating flow directions")
        
        if self.filled_heights is None:
            self.fill_depressions()
            
        n_cells = len(self.graph.points)
        self.flow_directions = np.full(n_cells, -1, dtype=np.int32)
        
        for cell_idx in range(n_cells):
            current_height = self.filled_heights[cell_idx]
            lowest_neighbor = -1
            lowest_height = current_height
            
            # Find lowest neighbor
            for neighbor_idx in self.graph.cell_neighbors[cell_idx]:
                neighbor_height = self.filled_heights[neighbor_idx]
                if neighbor_height < lowest_height:
                    lowest_height = neighbor_height
                    lowest_neighbor = neighbor_idx
                    
            # If we found a lower neighbor, flow to it
            if lowest_neighbor != -1:
                self.flow_directions[cell_idx] = lowest_neighbor
            # Otherwise, this is a sink (border cell or lake)
                
        logger.info("Flow directions calculated")
        
    def simulate_water_flow(self):
        """
        Simulate water flow accumulation using precipitation data.
        
        This calculates how much water flows through each cell.
        """
        logger.info("Simulating water flow")
        
        if self.flow_directions is None:
            self.calculate_flow_directions()
            
        n_cells = len(self.graph.points)
        self.water_flux = np.zeros(n_cells, dtype=np.float32)
        
        # Initialize with precipitation
        if hasattr(self.graph, 'precipitation'):
            self.water_flux = self.graph.precipitation.astype(np.float32).copy()
        else:
            # Default precipitation if not available
            self.water_flux.fill(10.0)
            
        # Process cells in topological order (from high to low)
        # Create list of (height, cell_index) and sort by height (descending)
        height_order = [(self.filled_heights[i], i) for i in range(n_cells)]
        height_order.sort(reverse=True)
        
        # Accumulate flow from high to low
        for _, cell_idx in height_order:
            flow_target = self.flow_directions[cell_idx]
            
            if flow_target != -1:  # Has downstream neighbor
                # Transfer water to downstream cell
                self.water_flux[flow_target] += self.water_flux[cell_idx]
                
        logger.info("Water flow simulation completed",
                   max_flow=np.max(self.water_flux),
                   total_flow=np.sum(self.water_flux))
        
    def generate_rivers(self):
        """
        Generate rivers based on water flow accumulation.
        
        Rivers form where water flow exceeds the minimum threshold.
        """
        logger.info("Generating rivers")
        
        if self.water_flux is None:
            self.simulate_water_flow()
            
        self.rivers = []
        river_cells = set()
        
        # Find cells with sufficient flow to form rivers
        river_candidates = []
        for i in range(len(self.graph.points)):
            if self.water_flux[i] >= self.options.min_river_flow:
                river_candidates.append((self.water_flux[i], i))
                
        # Sort by flow (descending) to process major rivers first
        river_candidates.sort(reverse=True)
        
        river_id = 0
        for flow, source_cell in river_candidates:
            if source_cell in river_cells:
                continue  # Already part of another river
                
            # Trace river path downstream
            river_path = self._trace_river_path(source_cell, river_cells)
            
            if len(river_path) > 1:  # Valid river
                river = River(
                    id=river_id,
                    cells=river_path,
                    flow=flow,
                    length=self._calculate_river_length(river_path),
                    source_cell=river_path[0],
                    mouth_cell=river_path[-1]
                )
                
                self.rivers.append(river)
                river_cells.update(river_path)
                river_id += 1
                
        logger.info("Rivers generated", count=len(self.rivers))
        
    def _trace_river_path(self, start_cell: int, existing_rivers: Set[int]) -> List[int]:
        """
        Trace a river path from source to mouth.
        
        Args:
            start_cell: Starting cell index
            existing_rivers: Set of cells already part of rivers
            
        Returns:
            List of cell indices forming the river path
        """
        path = [start_cell]
        current_cell = start_cell
        visited = {start_cell}
        
        while True:
            next_cell = self.flow_directions[current_cell]
            
            # Stop conditions
            if next_cell == -1:  # Reached sink
                break
            if next_cell in visited:  # Circular flow (shouldn't happen after depression filling)
                break
            if next_cell in existing_rivers:  # Reached another river
                path.append(next_cell)
                break
            if self.water_flux[next_cell] < self.options.min_river_flow:  # Flow too small
                break
                
            path.append(next_cell)
            visited.add(next_cell)
            current_cell = next_cell
            
        return path
        
    def _calculate_river_length(self, river_path: List[int]) -> float:
        """
        Calculate the length of a river path.
        
        Args:
            river_path: List of cell indices
            
        Returns:
            River length in map units
        """
        if len(river_path) < 2:
            return 0.0
            
        total_length = 0.0
        for i in range(len(river_path) - 1):
            p1 = self.graph.points[river_path[i]]
            p2 = self.graph.points[river_path[i + 1]]
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            total_length += distance
            
        return total_length
        
    def detect_lakes(self):
        """
        Detect lakes in closed basins.
        
        Lakes form in areas where water accumulates but cannot drain.
        """
        logger.info("Detecting lakes")
        
        if self.water_flux is None:
            self.simulate_water_flow()
            
        self.lakes = []
        processed_cells = set()
        
        # Find potential lake cells (high water accumulation, no outflow)
        lake_candidates = []
        for i in range(len(self.graph.points)):
            # Skip if already processed or is a river
            if i in processed_cells:
                continue
                
            # Check if this could be a lake cell
            if (self.water_flux[i] > self.options.min_river_flow * 0.5 and
                self.flow_directions[i] == -1):  # No outflow
                lake_candidates.append(i)
                
        # Group connected lake cells into lakes
        lake_id = 0
        for candidate_cell in lake_candidates:
            if candidate_cell in processed_cells:
                continue
                
            # Find all connected cells at similar water level
            lake_cells = self._find_lake_cells(candidate_cell, processed_cells)
            
            if len(lake_cells) >= 3:  # Minimum size for a lake
                water_level = np.mean([self.filled_heights[cell] for cell in lake_cells])
                
                lake = Lake(
                    id=lake_id,
                    cells=lake_cells,
                    water_level=water_level,
                    area=len(lake_cells) * (self.graph.spacing ** 2)
                )
                
                # Find outlet if any
                lake.outlet_cell = self._find_lake_outlet(lake_cells)
                
                self.lakes.append(lake)
                processed_cells.update(lake_cells)
                lake_id += 1
                
        logger.info("Lakes detected", count=len(self.lakes))
        
    def _find_lake_cells(self, start_cell: int, processed_cells: Set[int]) -> Set[int]:
        """
        Find all connected cells that form a lake.
        
        Args:
            start_cell: Starting cell for lake detection
            processed_cells: Set of already processed cells
            
        Returns:
            Set of cell indices forming the lake
        """
        lake_cells = set()
        queue = deque([start_cell])
        start_height = self.filled_heights[start_cell]
        
        while queue:
            cell = queue.popleft()
            
            if cell in lake_cells or cell in processed_cells:
                continue
                
            # Check if cell is at similar height and has water
            if (abs(self.filled_heights[cell] - start_height) < 2.0 and
                self.water_flux[cell] > 0):
                
                lake_cells.add(cell)
                
                # Add neighbors to queue
                for neighbor in self.graph.cell_neighbors[cell]:
                    if neighbor not in lake_cells and neighbor not in processed_cells:
                        queue.append(neighbor)
                        
        return lake_cells
        
    def _find_lake_outlet(self, lake_cells: Set[int]) -> Optional[int]:
        """
        Find the outlet cell for a lake.
        
        Args:
            lake_cells: Set of cells forming the lake
            
        Returns:
            Cell index of the outlet, or None if no outlet
        """
        outlet_candidates = []
        
        for cell in lake_cells:
            for neighbor in self.graph.cell_neighbors[cell]:
                if neighbor not in lake_cells:
                    # This is a potential outlet
                    height_diff = self.filled_heights[cell] - self.filled_heights[neighbor]
                    if height_diff > 0:  # Water can flow out
                        outlet_candidates.append((height_diff, cell))
                        
        if outlet_candidates:
            # Return the outlet with the largest height difference
            outlet_candidates.sort(reverse=True)
            return outlet_candidates[0][1]
            
        return None
        
    def run_full_simulation(self):
        """
        Run the complete hydrology simulation pipeline.
        
        This executes all steps in the correct order:
        1. Fill depressions
        2. Calculate flow directions
        3. Simulate water flow
        4. Generate rivers
        5. Detect lakes
        """
        logger.info("Starting full hydrology simulation")
        
        self.fill_depressions()
        self.calculate_flow_directions()
        self.simulate_water_flow()
        self.generate_rivers()
        self.detect_lakes()
        
        logger.info("Hydrology simulation completed",
                   rivers=len(self.rivers),
                   lakes=len(self.lakes))
        
        # Store results on graph for access by other modules
        self.graph.water_flux = self.water_flux
        self.graph.flow_directions = self.flow_directions
        self.graph.filled_heights = self.filled_heights
        self.graph.rivers = self.rivers
        self.graph.lakes = self.lakes



    def generate_rivers(self) -> Dict[int, RiverData]:
        """
        Generate river system following FMG's process.
        
        Returns:
            Dictionary of river data by river ID
        """
        logger.info("Starting river generation")

        # Step 1: Alter heights for water flow
        self.alter_heights()

        # Step 2: Resolve depressions iteratively
        self.resolve_depressions()

        # Step 3: Simulate water drainage and form rivers
        self.drain_water()

        # Step 4: Define final river properties
        self.define_rivers()

        logger.info(f"Generated {len(self.rivers)} rivers")
        return self.rivers

    def alter_heights(self) -> None:
        """
        Modify heightmap for water flow (Rivers.alterHeights()).
        
        This step prepares the terrain for realistic water flow by:
        - Storing original heights for reference
        - Making minor adjustments to eliminate flat areas
        """
        logger.info("Altering heights for water flow")

        # Store original heights
        self.original_heights = self.graph.heights.copy()

        # Check if distance field is available
        if not hasattr(self.graph, 'distance_field') or self.graph.distance_field is None:
            logger.warning("No distance field available, using fallback variation")
            # Fallback to small deterministic variation if distance field not available
            for i in range(len(self.graph.heights)):
                if self.graph.heights[i] >= self.options.sea_level:
                    variation = (hash(i) % 21 - 10) * 0.00001
                    self.graph.heights[i] += variation
            return

        # Add distance-based variations to break ties in flat areas (matches FMG exactly)
        # h + t[i] / 100 + d3.mean(c[i].map(c => t[c])) / 10000
        for i in range(len(self.graph.heights)):
            if self.graph.heights[i] >= self.options.sea_level:
                # Primary variation based on distance to water
                distance_variation = self.graph.distance_field[i] / 100.0

                # Secondary variation based on mean of neighbor distances
                neighbors = self._get_neighbors(i)
                if neighbors:
                    neighbor_distances = [self.graph.distance_field[n] for n in neighbors
                                        if n < len(self.graph.distance_field)]
                    mean_neighbor_distance = np.mean(neighbor_distances) if neighbor_distances else 0
                else:
                    mean_neighbor_distance = 0

                neighbor_variation = mean_neighbor_distance / 10000.0

                # Apply both variations
                self.graph.heights[i] += distance_variation + neighbor_variation

    def resolve_depressions(self) -> None:
        """
        Fill depressions iteratively to ensure proper water flow.
        
        This matches FMG's resolveDepressions function exactly.
        Processes cells from lowest to highest, raising cells that are lower
        than their lowest neighbor. Special handling for lakes.
        """
        logger.info("Resolving depressions")

        max_iterations = self.options.max_depression_iterations
        check_lake_max_iteration = int(max_iterations * 0.85)
        elevate_lake_max_iteration = int(max_iterations * 0.75)

        # Helper function to get height of lake or cell (matches FMG's height function)
        def height(i: int) -> float:
            if (hasattr(self.features, 'feature_ids') and
                self.features.feature_ids is not None and
                i < len(self.features.feature_ids)):
                feature_id = self.features.feature_ids[i]
                if feature_id > 0 and hasattr(self.features, 'features'):
                    for feature in self.features.features:
                        if feature and hasattr(feature, 'id') and feature.id == feature_id:
                            if hasattr(feature, 'height') and feature.height is not None:
                                return feature.height
            return self.graph.heights[i]

        # Get lakes and land cells
        lakes = []
        if hasattr(self.features, 'features'):
            lakes = [f for f in self.features.features if f and hasattr(f, 'type') and f.type == "lake"]

        # Get land cells excluding near-border cells
        land = []
        for i in range(len(self.graph.heights)):
            if self.graph.heights[i] >= self.options.sea_level and not self.graph.cell_border_flags[i]:
                land.append(i)

        # Sort land cells by height (lowest first)
        land.sort(key=lambda i: self.graph.heights[i])

        # Track progress for bad convergence detection
        progress = []
        depressions = float('inf')
        prev_depressions = None

        for iteration in range(max_iterations):
            # Check for bad progress (matches FMG logic)
            if len(progress) > 5 and sum(progress) > 0:
                # Bad progress, abort and set heights back
                self.alter_heights()  # Re-apply height alterations
                depressions = progress[0] if progress else 0
                logger.warning("Bad progress detected, reverting heights")
                break

            depressions = 0

            # Process lakes (only in early iterations)
            if iteration < check_lake_max_iteration:
                for lake in lakes:
                    if hasattr(lake, 'closed') and lake.closed:
                        continue

                    # Get lake shoreline cells
                    shoreline = []
                    if hasattr(lake, 'shoreline'):
                        shoreline = lake.shoreline
                    elif hasattr(self.features, 'feature_ids') and self.features.feature_ids is not None:
                        # Find shoreline cells - cells adjacent to this lake
                        for i in range(len(self.features.feature_ids)):
                            if self.features.feature_ids[i] == lake.id:
                                for neighbor in self._get_neighbors(i):
                                    if (neighbor < len(self.features.feature_ids) and
                                        self.features.feature_ids[neighbor] != lake.id and
                                        self.graph.heights[neighbor] >= self.options.sea_level):
                                        if neighbor not in shoreline:
                                            shoreline.append(neighbor)
                        lake.shoreline = shoreline

                    if not shoreline:
                        continue

                    # Find minimum shoreline height
                    min_height = min(self.graph.heights[s] for s in shoreline)

                    # Check if lake needs elevation
                    lake_height = lake.height if hasattr(lake, 'height') and lake.height is not None else 0
                    if min_height >= 100 or lake_height > min_height:
                        continue

                    # Handle lake elevation or closure
                    if iteration > elevate_lake_max_iteration:
                        # Restore original heights and close lake
                        for i in shoreline:
                            if self.original_heights is not None:
                                self.graph.heights[i] = self.original_heights[i]
                        lake.height = min(self.graph.heights[s] for s in shoreline) - 1
                        lake.closed = True
                        continue

                    depressions += 1
                    lake.height = min_height + 0.2

            # Process land cells
            for i in land:
                # Get minimum neighbor height (using height function for lakes)
                neighbor_heights = [height(c) for c in self._get_neighbors(i)]
                if not neighbor_heights:
                    continue

                min_height = min(neighbor_heights)

                # Check if cell is depressed (lower than lowest neighbor)
                if min_height >= 100 or self.graph.heights[i] > min_height:
                    continue

                depressions += 1
                self.graph.heights[i] = min_height + 0.1

            # Track progress
            if prev_depressions is not None:
                progress.append(depressions - prev_depressions)
            prev_depressions = depressions

            # Check if converged
            if depressions == 0:
                logger.info(f"Depression resolution converged after {iteration + 1} iterations")
                break

        if depressions > 0:
            logger.warning(f"Unresolved depressions: {depressions}. Edit heightmap to fix")


    def _get_min_neighbor_height(self, cell_id: int) -> float:
        """Get the minimum height among neighbors."""
        neighbors = self._get_neighbors(cell_id)
        if not neighbors:
            return self.graph.heights[cell_id]

        return min(self.graph.heights[neighbor_id] for neighbor_id in neighbors)

    def _get_neighbors(self, cell_id: int) -> List[int]:
        """Get neighbor cell IDs from VoronoiGraph cell connectivity."""
        if cell_id >= len(self.graph.cell_neighbors):
            return []

        return self.graph.cell_neighbors[cell_id]

    def _is_lake_cell(self, cell_id: int) -> bool:
        """Check if cell is part of a lake feature."""
        if not hasattr(self.features, 'features'):
            return False

        for feature in self.features.features:
            if feature and feature.type == "lake":
                # Check if cell_id is part of this lake feature
                if hasattr(self.features, 'feature_ids') and self.features.feature_ids is not None:
                    if cell_id < len(self.features.feature_ids) and self.features.feature_ids[cell_id] == feature.id:
                        return True
        return False

    def drain_water(self) -> None:
        """
        Simulate water drainage and river formation.
        
        This implements the core water flow algorithm matching FMG exactly:
        1. Pre-calculate lake outlets (like Lakes.defineClimateData)
        2. Process each land cell from highest to lowest:
           - Add precipitation flux
           - If cell is a lake outlet, add lake excess water
           - Flow water downhill, creating rivers where flux exceeds threshold
        """
        logger.info("Simulating water drainage")

        # Step 1: Pre-calculate lake outlets (equivalent to Lakes.defineClimateData)
        lake_out_cells = self._define_lake_climate_data()

        # Step 2: Process land cells and flow water downhill
        self._flow_water_downhill(lake_out_cells)


    def _define_lake_climate_data(self) -> Dict[int, List]:
        """
        Pre-calculate lake outlets and climate data (equivalent to Lakes.defineClimateData).
        
        Returns:
            Dictionary mapping outlet cell IDs to list of lakes that drain through them
        """
        lake_out_cells = {}

        if not hasattr(self.features, 'features'):
            return lake_out_cells

        # Process each lake feature
        for feature in self.features.features:
            if not feature or not hasattr(feature, 'type') or feature.type != "lake":
                continue

            # Get lake cells
            lake_cells = []
            if hasattr(self.features, 'feature_ids') and self.features.feature_ids is not None:
                for i, fid in enumerate(self.features.feature_ids):
                    if fid == feature.id:
                        lake_cells.append(i)

            if not lake_cells:
                continue

            # Calculate lake properties
            # Note: In FMG, flux is accumulated during the main loop, but we pre-calculate area/evaporation
            lake_area = len(lake_cells)
            feature.area = lake_area
            feature.evaporation = lake_area * 2.0  # Simplified evaporation rate
            feature.flux = 0  # Will be accumulated during main loop

            # Find outlet cell
            outlet_cell = self._find_lake_outlet(feature, lake_cells)
            if outlet_cell is not None:
                feature.outCell = outlet_cell
                # Map outlet cell to lakes that drain through it
                if outlet_cell not in lake_out_cells:
                    lake_out_cells[outlet_cell] = []
                lake_out_cells[outlet_cell].append(feature)

        return lake_out_cells

    def _find_lake_outlet(self, lake_feature, lake_cells: List[int]) -> Optional[int]:
        """Find the lowest point on lake perimeter for outlet."""
        if not lake_cells:
            return None

        # Find perimeter cells (lake cells with non-lake neighbors)
        perimeter_cells = []
        for cell_id in lake_cells:
            neighbors = self._get_neighbors(cell_id)
            for neighbor_id in neighbors:
                # If neighbor is not part of this lake, current cell is on perimeter
                if (not hasattr(self.features, 'feature_ids') or
                    self.features.feature_ids is None or
                    neighbor_id >= len(self.features.feature_ids) or
                    self.features.feature_ids[neighbor_id] != lake_feature.id):
                    perimeter_cells.append(cell_id)
                    break

        if not perimeter_cells:
            return None

        # Find lowest perimeter cell
        lowest_height = float('inf')
        outlet_cell = None

        for cell_id in perimeter_cells:
            if self.graph.heights[cell_id] < lowest_height:
                lowest_height = self.graph.heights[cell_id]
                outlet_cell = cell_id

        return outlet_cell

    def _flow_water_downhill(self, lake_out_cells: Dict[int, List]) -> None:
        """
        Flow water downhill, creating rivers where flux exceeds threshold.
        Integrates lake outlet processing during the main loop (matches FMG).
        """
        # Calculate cells number modifier for precipitation scaling
        cells_number_modifier = (len(self.graph.points) / 10000) ** 0.25

        # Process land cells in height order (highest first) - matches FMG exactly
        land_cells = []
        for i in range(len(self.graph.heights)):
            if self.graph.heights[i] >= self.options.sea_level:
                land_cells.append(i)

        # Sort by height - highest first (matches FMG's land.sort((a, b) => h[b] - h[a]))
        land_cells.sort(key=lambda i: self.graph.heights[i], reverse=True)

        for cell_id in land_cells:
            # Step 1: Add precipitation flux to this cell
            if (hasattr(self.climate, 'precipitation') and
                hasattr(self.graph, 'grid_indices') and
                self.graph.grid_indices is not None):
                # Use grid mapping to access original climate data
                grid_cell_id = self.graph.grid_indices[cell_id]
                if isinstance(self.climate.precipitation, dict):
                    precip = self.climate.precipitation.get(grid_cell_id, 50.0)
                else:
                    precip = self.climate.precipitation[grid_cell_id] if grid_cell_id < len(self.climate.precipitation) else 50.0
            else:
                # Fallback
                if isinstance(self.climate.precipitation, dict):
                    precip = self.climate.precipitation.get(cell_id, 50.0)
                else:
                    precip = self.climate.precipitation[cell_id] if cell_id < len(self.climate.precipitation) else 50.0

            self.flux[cell_id] += precip / cells_number_modifier

            # Step 2: Check if this cell is a lake outlet
            if cell_id in lake_out_cells:
                # Process each lake that drains through this outlet
                lakes = lake_out_cells[cell_id]
                for lake in lakes:
                    # Only process if lake flux exceeds evaporation
                    if lake.flux > lake.evaporation:
                        # Find the lake cell adjacent to this outlet
                        lake_cell = None
                        for neighbor_id in self._get_neighbors(cell_id):
                            if (neighbor_id < len(self.graph.heights) and
                                self.graph.heights[neighbor_id] < self.options.sea_level and
                                hasattr(self.features, 'feature_ids') and
                                self.features.feature_ids is not None and
                                neighbor_id < len(self.features.feature_ids) and
                                self.features.feature_ids[neighbor_id] == lake.id):
                                lake_cell = neighbor_id
                                break

                        if lake_cell is not None:
                            # Add excess lake water to the lake cell
                            excess_water = max(lake.flux - lake.evaporation, 0)
                            self.flux[lake_cell] += excess_water

                            # Handle river creation/assignment for lake (matches FMG logic)
                            if self.river_ids[lake_cell] != 0:
                                # Check if we should keep existing river identity
                                lake_river = self.river_ids[lake_cell]
                                same_river = any(
                                    self.river_ids[n] == lake_river
                                    for n in self._get_neighbors(lake_cell)
                                    if n < len(self.river_ids)
                                )

                                if not same_river:
                                    # Create new river for lake
                                    self.river_ids[lake_cell] = self.next_river_id
                                    self.rivers[self.next_river_id] = RiverData(id=self.next_river_id)
                                    self.rivers[self.next_river_id].cells.append(lake_cell)
                                    self.next_river_id += 1
                            else:
                                # Create new river for lake
                                self.river_ids[lake_cell] = self.next_river_id
                                self.rivers[self.next_river_id] = RiverData(id=self.next_river_id)
                                self.rivers[self.next_river_id].cells.append(lake_cell)
                                self.next_river_id += 1

                            # Set lake outlet river
                            lake.outlet = self.river_ids[lake_cell]

                            # Flow lake water downstream
                            self._flow_down(cell_id, self.flux[lake_cell], lake.outlet)

                # Handle tributary assignment (matches FMG)
                if lakes:
                    outlet = lakes[0].outlet if hasattr(lakes[0], 'outlet') else None
                    for lake in lakes:
                        if hasattr(lake, 'inlets') and isinstance(lake.inlets, list):
                            for inlet in lake.inlets:
                                if inlet in self.rivers and outlet:
                                    self.rivers[inlet].parent_id = outlet

            # Step 3: Handle near-border cells
            if self.graph.cell_border_flags[cell_id] and self.river_ids[cell_id] > 0:
                # Add border cell (-1) to river
                if self.river_ids[cell_id] in self.rivers:
                    self.rivers[self.river_ids[cell_id]].cells.append(-1)
                continue

            # Step 4: Find downhill flow target
            # Special handling for lake outlet cells - exclude lake cells from targets
            if cell_id in lake_out_cells:
                # Get all lake feature IDs for this outlet
                lake_ids = [lake.id for lake in lake_out_cells[cell_id]]
                target_cell = self._find_flow_target_excluding_lakes(cell_id, lake_ids)
            else:
                target_cell = self._find_flow_target(cell_id)

            if target_cell is None:
                continue  # No downhill flow possible

            # Check if cell is actually depressed (FMG logic)
            if self.graph.heights[cell_id] <= self.graph.heights[target_cell]:
                continue

            # Step 5: Handle flux transfer based on amount
            cell_flux = self.flux[cell_id]

            if cell_flux < self.options.min_river_flux:
                # Below river threshold - just transfer flux
                if self.graph.heights[target_cell] >= self.options.sea_level:
                    self.flux[target_cell] += cell_flux
                continue

            # Above river threshold - create/extend river
            if self.river_ids[cell_id] == 0:
                # Create new river
                river_id = self.next_river_id
                self.next_river_id += 1
                self.rivers[river_id] = RiverData(id=river_id)
                self.river_ids[cell_id] = river_id
                self.rivers[river_id].cells.append(cell_id)
            else:
                river_id = self.river_ids[cell_id]

            # Flow water downstream using FMG's flowDown logic
            self._flow_down(target_cell, cell_flux, river_id)

    def _find_flow_target(self, cell_id: int) -> Optional[int]:
        """Find the lowest neighbor to flow water to."""
        neighbors = self._get_neighbors(cell_id)
        if not neighbors:
            return None

        lowest_height = self.graph.heights[cell_id]
        target_cell = None

        for neighbor_id in neighbors:
            neighbor_height = self.graph.heights[neighbor_id]
            if neighbor_height < lowest_height:
                lowest_height = neighbor_height
                target_cell = neighbor_id

        return target_cell

    def _find_flow_target_excluding_lakes(self, cell_id: int, lake_ids: List[int]) -> Optional[int]:
        """Find the lowest neighbor to flow water to, excluding cells in specified lakes."""
        neighbors = self._get_neighbors(cell_id)
        if not neighbors:
            return None

        # Filter out neighbors that belong to any of the specified lakes
        filtered_neighbors = []
        for neighbor_id in neighbors:
            # Check if neighbor belongs to any excluded lake
            in_excluded_lake = False
            if (hasattr(self.features, 'feature_ids') and
                self.features.feature_ids is not None and
                neighbor_id < len(self.features.feature_ids)):
                feature_id = self.features.feature_ids[neighbor_id]
                if feature_id in lake_ids:
                    in_excluded_lake = True

            if not in_excluded_lake:
                filtered_neighbors.append(neighbor_id)

        if not filtered_neighbors:
            return None

        # Find lowest among filtered neighbors
        lowest_height = self.graph.heights[cell_id]
        target_cell = None

        for neighbor_id in filtered_neighbors:
            neighbor_height = self.graph.heights[neighbor_id]
            if neighbor_height < lowest_height:
                lowest_height = neighbor_height
                target_cell = neighbor_id

        return target_cell

    def _create_or_extend_river(self, from_cell: int, to_cell: int) -> None:
        """Create new river or extend existing one."""
        from_river_id = self.river_ids[from_cell]
        to_river_id = self.river_ids[to_cell]

        if from_river_id == 0 and to_river_id == 0:
            # Create new river
            river_id = self.next_river_id
            self.next_river_id += 1

            self.rivers[river_id] = RiverData(id=river_id)
            self.river_ids[from_cell] = river_id
            self.rivers[river_id].cells.append(from_cell)

        elif from_river_id > 0 and to_river_id == 0:
            # Extend existing river
            self.river_ids[to_cell] = from_river_id
            self.rivers[from_river_id].cells.append(to_cell)

        elif from_river_id == 0 and to_river_id > 0:
            # Join existing river
            self.river_ids[from_cell] = to_river_id
            self.rivers[to_river_id].cells.append(from_cell)

        elif from_river_id > 0 and to_river_id > 0 and from_river_id != to_river_id:
            # River confluence - merge based on flux
            from_flux = self.flux[from_cell]
            to_flux = self.flux[to_cell]

            if from_flux > to_flux:
                # from_river takes over
                self._merge_rivers(to_river_id, from_river_id)
                self.confluences[to_cell] = True
            else:
                # to_river takes over
                self._merge_rivers(from_river_id, to_river_id)
                self.confluences[from_cell] = True

    def _merge_rivers(self, tributary_id: int, main_river_id: int) -> None:
        """Merge tributary river into main river."""
        if tributary_id not in self.rivers or main_river_id not in self.rivers:
            return

        tributary = self.rivers[tributary_id]
        tributary.parent_id = main_river_id

        # Update cell assignments
        for cell_id in tributary.cells:
            self.river_ids[cell_id] = main_river_id
            self.rivers[main_river_id].cells.append(cell_id)

    def _flow_down(self, to_cell: int, from_flux: float, river_id: int) -> None:
        """Transfer flux downstream following FMG's flowDown algorithm exactly."""
        # Get current flux and river for target cell
        to_flux = self.flux[to_cell] - self.confluences[to_cell].astype(float).sum() if hasattr(self.confluences[to_cell], 'sum') else (self.flux[to_cell] - (1.0 if self.confluences[to_cell] else 0.0))
        to_river_id = self.river_ids[to_cell]

        if to_river_id > 0:
            # Handle river confluence - FMG logic
            if from_flux > to_flux:
                # Incoming river is larger - takes over
                self.confluences[to_cell] = True
                # Set tributary relationship
                if to_river_id in self.rivers:
                    self.rivers[to_river_id].parent_id = river_id
                # Reassign cell to larger river
                self.river_ids[to_cell] = river_id
            else:
                # Existing river is larger - incoming becomes tributary
                self.confluences[to_cell] = True
                if river_id in self.rivers:
                    self.rivers[river_id].parent_id = to_river_id
        else:
            # Assign river to new cell
            self.river_ids[to_cell] = river_id

        # CRITICAL: Accumulate flux downstream if on land
        if self.graph.heights[to_cell] >= self.options.sea_level:
            self.flux[to_cell] += from_flux
        else:
            # Pour water to water body (lake or ocean)
            if hasattr(self.features, 'feature_ids') and self.features.feature_ids is not None:
                if to_cell < len(self.features.feature_ids):
                    feature_id = self.features.feature_ids[to_cell]
                    if feature_id > 0 and hasattr(self.features, 'features'):
                        # Find the feature
                        for feature in self.features.features:
                            if feature and hasattr(feature, 'id') and feature.id == feature_id:
                                if hasattr(feature, 'type') and feature.type == "lake":
                                    # Update lake properties when river flows into it
                                    if not hasattr(feature, 'river') or from_flux > getattr(feature, 'enteringFlux', 0):
                                        feature.river = river_id
                                        feature.enteringFlux = from_flux
                                    feature.flux = getattr(feature, 'flux', 0) + from_flux
                                    if not hasattr(feature, 'inlets'):
                                        feature.inlets = []
                                    if river_id not in feature.inlets:
                                        feature.inlets.append(river_id)
                                break

        # Add cell to river
        if river_id in self.rivers:
            self.rivers[river_id].cells.append(to_cell)

    def define_rivers(self) -> None:
        """Define final river properties including width, length, and discharge."""
        logger.info("Defining river properties")

        # Filter out tiny rivers (less than 3 cells) to match FMG
        rivers_to_remove = []
        for river_id, river in self.rivers.items():
            if len(river.cells) < 3:
                rivers_to_remove.append(river_id)

        # Remove tiny rivers
        for river_id in rivers_to_remove:
            del self.rivers[river_id]

        logger.info(f"Filtered out {len(rivers_to_remove)} tiny rivers")

        for river_id, river in self.rivers.items():
            if not river.cells:
                continue

            # Calculate discharge (final flux at river mouth)
            mouth_cell = river.cells[-1]
            river.discharge = self.flux[mouth_cell]

            # Calculate width based on discharge and slope
            river.width = self._calculate_river_width(river.discharge, river.cells)

            # Calculate approximate length
            river.length = self._calculate_river_length(river.cells)

            # Calculate distance from source for the mouth
            if river.cells:
                river.source_distance = self._calculate_source_distance(river.cells)

    def _calculate_river_width(self, discharge: float, river_cells: Optional[List[int]] = None) -> float:
        """
        Calculate river width using hydraulic formulas.
        
        Uses a combination of:
        1. Empirical width-discharge relationship
        2. Manning's equation considerations for slope effects
        3. Channel geometry assumptions
        
        Args:
            discharge: River discharge (flow rate)
            river_cells: Optional list of river cells for slope calculation
            
        Returns:
            River width in map units
        """
        if discharge <= 0:
            return 0.0

        # Calculate average slope if river cells are provided
        slope = self.options.min_slope  # Default minimum slope
        if river_cells and len(river_cells) >= 2:
            slope = max(self._calculate_average_slope(river_cells), self.options.min_slope)

        # Empirical width-discharge relationship (Leopold & Maddock, 1953)
        # W = a * Q^b, where a ≈ 2.3, b ≈ 0.5 for natural channels
        empirical_width = 2.3 * (discharge ** 0.5)

        # Adjust for slope using Manning's equation principles
        # Steeper slopes → narrower, deeper channels
        # Gentler slopes → wider, shallower channels
        slope_factor = (self.options.min_slope / slope) ** 0.2  # Gentle adjustment

        # Apply roughness coefficient influence
        # Higher roughness → wider channels to maintain flow
        roughness_factor = (self.options.manning_n / 0.035) ** 0.1

        # Combine factors
        hydraulic_width = empirical_width * slope_factor * roughness_factor

        # Apply original scale factor for compatibility
        final_width = hydraulic_width * self.options.width_scale_factor

        return max(final_width, 1.0)  # Minimum width of 1

    def _calculate_average_slope(self, river_cells: List[int]) -> float:
        """
        Calculate average slope along a river path.
        
        Args:
            river_cells: List of cell IDs forming the river path
            
        Returns:
            Average slope as elevation change per unit distance
        """
        if len(river_cells) < 2:
            return self.options.min_slope

        total_elevation_drop = 0.0
        total_distance = 0.0

        for i in range(len(river_cells) - 1):
            cell1 = river_cells[i]
            cell2 = river_cells[i + 1]

            # Calculate elevation difference
            elev1 = self.graph.heights[cell1]
            elev2 = self.graph.heights[cell2]
            elevation_drop = abs(elev1 - elev2)

            # Calculate distance between cells
            point1 = self.graph.points[cell1]
            point2 = self.graph.points[cell2]
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

            if distance > 0:
                total_elevation_drop += elevation_drop
                total_distance += distance

        if total_distance > 0:
            return total_elevation_drop / total_distance
        else:
            return self.options.min_slope

    def _calculate_river_length(self, cells: List[int]) -> float:
        """Calculate approximate river length with meandering."""
        if len(cells) < 2:
            return 0.0

        total_length = 0.0

        for i in range(len(cells) - 1):
            cell1 = cells[i]
            cell2 = cells[i + 1]

            # Calculate Euclidean distance between cell centers
            p1 = self.graph.points[cell1]
            p2 = self.graph.points[cell2]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            # Apply meandering factor
            meandered_length = segment_length * (1.0 + self.options.meandering_factor)
            total_length += meandered_length

        return total_length

    def _calculate_source_distance(self, cells: List[int]) -> float:
        """Calculate distance from source to mouth."""
        if len(cells) < 2:
            return 0.0

        source = self.graph.points[cells[0]]
        mouth = self.graph.points[cells[-1]]

        return math.sqrt((mouth[0] - source[0])**2 + (mouth[1] - source[1])**2)

