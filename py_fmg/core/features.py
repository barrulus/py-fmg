"""
Geographic features detection and markup.

This module handles:
- Ocean/land classification based on height
- Lake detection in deep depressions
- Coastline detection
- Island identification
- Distance field calculation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Feature type constants
DEEPER_LAND = 3
LANDLOCKED = 2
LAND_COAST = 1
UNMARKED = 0
WATER_COAST = -1
DEEP_WATER = -2


@dataclass
class Feature:
    """Represents a geographic feature (ocean, lake, island)."""
    id: int
    type: str  # "ocean", "lake", "island"
    land: bool
    border: bool  # touches map edge
    cells: int  # total cells in feature
    first_cell: int
    vertices: List[int] = None
    area: float = 0.0
    height: Optional[float] = None  # for lakes
    shoreline: Optional[List[int]] = None  # for lakes


class Features:
    """Handles geographic feature detection and markup."""
    
    def __init__(self, graph, seed: Optional[str] = None):
        """
        Initialize Features with a VoronoiGraph.
        
        Args:
            graph: VoronoiGraph instance with populated heights
            seed: Optional seed for PRNG reseeding
        """
        self.graph = graph
        self.n_cells = len(graph.points)
        
        # Use border_cells if available, otherwise use cell_border_flags
        if hasattr(graph, 'border_cells') and graph.border_cells is not None:
            self.border_cells = graph.border_cells
        else:
            self.border_cells = graph.cell_border_flags
        
        # Reseed PRNG if seed provided (matches FMG's features.js:31)
        if seed:
            from ..utils.random import set_random_seed
            set_random_seed(seed)
        
    def is_land(self, cell_id: int) -> bool:
        """Check if a cell is land (height >= 20)."""
        return self.graph.heights[cell_id] >= 20
    
    def is_water(self, cell_id: int) -> bool:
        """Check if a cell is water (height < 20)."""
        return self.graph.heights[cell_id] < 20
    
    def markup_grid(self):
        """
        Mark grid features (ocean, lakes, islands) and calculate distance field.
        
        This is the equivalent of FMG's Features.markupGrid()
        """
        # Initialize arrays
        self.distance_field = np.zeros(self.n_cells, dtype=np.int8)
        self.feature_ids = np.zeros(self.n_cells, dtype=np.uint16)
        self.features = [None]  # index 0 is reserved
        
        # Use BFS to identify connected features
        # Find the cell with the lowest height value (guaranteed to be ocean)
        guaranteed_ocean_cell = np.argmin(self.graph.heights)
        queue = deque([guaranteed_ocean_cell])
        feature_id = 1
        
        while len(queue) > 0 and queue[0] != -1:
            first_cell = queue[0]
            self.feature_ids[first_cell] = feature_id
            
            land = self.is_land(first_cell)
            border = False  # set true if feature touches map edge
            cell_count = 0
            
            # BFS to mark all cells in this feature
            while queue:
                cell_id = queue.popleft()
                cell_count += 1
                
                # Check if on border
                if not border and self.border_cells[cell_id]:
                    border = True
                
                # Check all neighbors
                for neighbor_id in self.graph.cell_neighbors[cell_id]:
                    is_neib_land = self.is_land(neighbor_id)
                    
                    # If same type (land/water) and unmarked, add to feature
                    if land == is_neib_land and self.feature_ids[neighbor_id] == UNMARKED:
                        self.feature_ids[neighbor_id] = feature_id
                        queue.append(neighbor_id)
                    # Mark coastline cells
                    elif land and not is_neib_land:
                        self.distance_field[cell_id] = LAND_COAST
                        self.distance_field[neighbor_id] = WATER_COAST
            
            # Determine feature type
            if land:
                feature_type = "island"
            elif border:
                feature_type = "ocean"
            else:
                feature_type = "lake"
            
            # Create feature
            feature = Feature(
                id=feature_id,
                type=feature_type,
                land=land,
                border=border,
                cells=cell_count,
                first_cell=first_cell
            )
            self.features.append(feature)
            
            # Find next unmarked cell
            try:
                next_unmarked = next(i for i, f in enumerate(self.feature_ids) if f == UNMARKED)
                queue.append(next_unmarked)
            except StopIteration:
                queue.append(-1)  # No more unmarked cells
            
            feature_id += 1
        
        # Markup deep ocean cells
        self._markup_distance_field(
            start=DEEP_WATER, 
            increment=-1, 
            limit=-10
        )
        
        # Store results
        self.graph.distance_field = self.distance_field
        self.graph.feature_ids = self.feature_ids
        self.graph.features = self.features
    
    def _markup_distance_field(self, start: int, increment: int, limit: int = 127):
        """
        Calculate distance field for cells.
        
        Args:
            start: Starting distance value
            increment: Distance increment per iteration
            limit: Maximum distance to calculate
        """
        distance = start
        
        while True:
            marked = 0
            prev_distance = distance - increment
            
            # Find all cells at previous distance
            for cell_id in range(self.n_cells):
                if self.distance_field[cell_id] != prev_distance:
                    continue
                
                # Mark unmarked neighbors
                for neighbor_id in self.graph.cell_neighbors[cell_id]:
                    if self.distance_field[neighbor_id] == UNMARKED:
                        self.distance_field[neighbor_id] = distance
                        marked += 1
            
            # Stop if no cells marked or limit reached
            if marked == 0 or distance == limit:
                break
            
            distance += increment
    
    def add_lakes_in_deep_depressions(self, elevation_limit: float = 20):
        """
        Add lakes in deep depressions that cannot drain to ocean.
        
        Args:
            elevation_limit: Maximum elevation difference for water to flow
        """
        if elevation_limit == 80:
            return  # Disabled
        
        for i in range(self.n_cells):
            # Skip border cells and water
            if self.border_cells[i] or self.graph.heights[i] < 20:
                continue
            
            # Check if this is a local minimum
            neighbor_heights = [self.graph.heights[n] for n in self.graph.cell_neighbors[i]]
            min_height = min(neighbor_heights) if neighbor_heights else float('inf')
            
            if self.graph.heights[i] > min_height:
                continue
            
            # Check if water can flow to ocean
            deep = True
            threshold = self.graph.heights[i] + elevation_limit
            queue = deque([i])
            checked = set([i])
            
            while deep and queue:
                q = queue.popleft()
                
                for n in self.graph.cell_neighbors[q]:
                    if n in checked:
                        continue
                    if self.graph.heights[n] >= threshold:
                        continue
                    if self.graph.heights[n] < 20:
                        deep = False
                        break
                    
                    checked.add(n)
                    queue.append(n)
            
            # If water cannot flow out, add a lake
            if deep:
                # Include cell and neighbors at same height
                lake_cells = [i]
                for n in self.graph.cell_neighbors[i]:
                    if self.graph.heights[n] == self.graph.heights[i]:
                        lake_cells.append(n)
                
                self._add_lake(lake_cells)
    
    def _add_lake(self, lake_cells: List[int]):
        """Add a lake feature from given cells."""
        feature_id = len(self.features)
        
        # Convert cells to water
        for cell_id in lake_cells:
            self.graph.heights[cell_id] = 19
            self.distance_field[cell_id] = WATER_COAST
            self.feature_ids[cell_id] = feature_id
            
            # Mark neighbors as coastline
            for n in self.graph.cell_neighbors[cell_id]:
                if n not in lake_cells and self.graph.heights[n] >= 20:
                    self.distance_field[n] = LAND_COAST
        
        # Add lake feature
        feature = Feature(
            id=feature_id,
            type="lake",
            land=False,
            border=False,
            cells=len(lake_cells),
            first_cell=lake_cells[0]
        )
        self.features.append(feature)
    
    def open_near_sea_lakes(self, breach_limit: float = 22):
        """
        Open lakes that are near the ocean and separated by low elevation.
        
        Args:
            breach_limit: Maximum height that can be breached by water
        """
        # Skip if no lakes
        if not any(f.type == "lake" for f in self.features if f):
            return
        
        for i in range(self.n_cells):
            lake_feature_id = self.feature_ids[i]
            
            # Skip if not a lake
            if (lake_feature_id >= len(self.features) or 
                not self.features[lake_feature_id] or
                self.features[lake_feature_id].type != "lake"):
                continue
            
            # Check neighbors
            for c in self.graph.cell_neighbors[i]:
                # Check if this is a low coastline cell
                if self.distance_field[c] != LAND_COAST or self.graph.heights[c] > breach_limit:
                    continue
                
                # Check if neighbor touches ocean
                for n in self.graph.cell_neighbors[c]:
                    ocean_id = self.feature_ids[n]
                    if (ocean_id < len(self.features) and 
                        self.features[ocean_id] and
                        self.features[ocean_id].type == "ocean"):
                        self._remove_lake(c, lake_feature_id, ocean_id)
                        break
                else:
                    continue
                break
    
    def _remove_lake(self, threshold_cell: int, lake_id: int, ocean_id: int):
        """Convert a lake to ocean by breaching at threshold cell."""
        # Convert threshold cell to water
        self.graph.heights[threshold_cell] = 19
        self.distance_field[threshold_cell] = WATER_COAST
        self.feature_ids[threshold_cell] = ocean_id
        
        # Mark neighbors as coastline
        for c in self.graph.cell_neighbors[threshold_cell]:
            if self.graph.heights[c] >= 20:
                self.distance_field[c] = LAND_COAST
        
        # Convert all lake cells to ocean
        for i in range(self.n_cells):
            if self.feature_ids[i] == lake_id:
                self.feature_ids[i] = ocean_id
        
        # Mark former lake as ocean
        if self.features[lake_id]:
            self.features[lake_id].type = "ocean"