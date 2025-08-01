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

