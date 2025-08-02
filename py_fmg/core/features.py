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
from typing import List, Tuple, Optional
from dataclasses import dataclass

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
        if hasattr(graph, "border_cells") and graph.border_cells is not None:
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

        # Use DFS to identify connected features (matching FMG's behavior)
        # Start from cell 0 for FMG compatibility
        queue = [0]
        feature_id = 1

        while len(queue) > 0 and queue[0] != -1:
            first_cell = queue[0]
            self.feature_ids[first_cell] = feature_id

            land = self.is_land(first_cell)
            border = False  # set true if feature touches map edge
            cell_count = 0

            # DFS to mark all cells in this feature (using pop() like FMG)
            while queue:
                cell_id = queue.pop()
                cell_count += 1

                # Check if on border
                if not border and self.border_cells[cell_id]:
                    border = True

                # Check all neighbors
                for neighbor_id in self.graph.cell_neighbors[cell_id]:
                    is_neib_land = self.is_land(neighbor_id)

                    # If same type (land/water) and unmarked, add to feature
                    if (
                        land == is_neib_land
                        and self.feature_ids[neighbor_id] == UNMARKED
                    ):
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
                first_cell=first_cell,
            )
            self.features.append(feature)

            # Find next unmarked cell
            try:
                next_unmarked = next(
                    i for i, f in enumerate(self.feature_ids) if f == UNMARKED
                )
                queue.append(next_unmarked)
            except StopIteration:
                queue.append(-1)  # No more unmarked cells

            feature_id += 1

        # Markup deep ocean cells
        self._markup_distance_field(start=DEEP_WATER, increment=-1, limit=-10)

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

        # Track which cells have already been processed to avoid duplicate lakes
        processed = set()

        for i in range(self.n_cells):
            # Skip border cells, water, and already processed cells
            if self.border_cells[i] or self.graph.heights[i] < 20 or i in processed:
                continue

            # Check if this is a local minimum (lower than at least one neighbor)
            neighbor_heights = [
                self.graph.heights[n] for n in self.graph.cell_neighbors[i]
            ]
            min_height = min(neighbor_heights) if neighbor_heights else float("inf")
            max_height = max(neighbor_heights) if neighbor_heights else float("-inf")

            # Skip if not a depression (must be surrounded by higher ground)
            if self.graph.heights[i] >= max_height:
                continue

            # Check if water can flow to ocean
            deep = True
            threshold = self.graph.heights[i] + elevation_limit
            queue = [i]
            checked = set([i])

            while deep and queue:
                q = queue.pop()

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
                # Fill the entire depression up to the threshold height
                # The 'checked' set contains all cells explored during the water flow test
                # These are all cells that water would fill in the depression
                lake_cells = list(checked)

                # Mark all lake cells as processed to avoid creating duplicate lakes
                processed.update(lake_cells)

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
            first_cell=lake_cells[0],
        )
        self.features.append(feature)

    def open_near_sea_lakes(self, breach_limit: float = 22):
        """
        Opens lakes near the ocean. This version safely separates finding from acting.
        """
        ocean_ids = {f.id for f in self.features if f and f.type == "ocean"}
        if not ocean_ids:
            return

        breach_actions = []  # A list of (breach_cell, lake_id, ocean_id) tuples

        # Use a copy of the list to iterate over, to be safe.
        lakes_to_check = [f for f in self.features if f and f.type == "lake"]

        for lake in lakes_to_check:
            lake_cells = [i for i, fid in enumerate(self.feature_ids) if fid == lake.id]

            found_breach_for_this_lake = False
            for lake_cell in lake_cells:
                for coast_candidate in self.graph.cell_neighbors[lake_cell]:
                    is_coast = self.distance_field[coast_candidate] == LAND_COAST
                    is_low = self.graph.heights[coast_candidate] <= breach_limit

                    if is_coast and is_low:
                        for ocean_neighbor in self.graph.cell_neighbors[
                            coast_candidate
                        ]:
                            if self.feature_ids[ocean_neighbor] in ocean_ids:
                                # Found a breach. Record the action and stop searching for this lake.
                                action = (
                                    coast_candidate,
                                    lake.id,
                                    self.feature_ids[ocean_neighbor],
                                )
                                breach_actions.append(action)
                                found_breach_for_this_lake = True
                                break
                        if found_breach_for_this_lake:
                            break
                if found_breach_for_this_lake:
                    break

        # Now, execute all the recorded breach actions.
        # This ensures we don't modify the feature list while iterating over it.
        for breach_cell, lake_id, ocean_id in breach_actions:
            self._remove_lake(breach_cell, lake_id, ocean_id)

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

    def markup_pack(self, packed_graph):
        """
        Mark packed features (ocean, lakes, islands) and calculate additional properties.

        This is the equivalent of FMG's Features.markupPack()

        Args:
            packed_graph: Packed VoronoiGraph from regraph()
        """
        n_cells = len(packed_graph.points)
        if n_cells == 0:
            return  # No cells to process

        # Initialize arrays
        distance_field = np.zeros(n_cells, dtype=np.int8)
        feature_ids = np.zeros(n_cells, dtype=np.uint16)
        haven = np.zeros(
            n_cells, dtype=np.uint16
        )  # Opposite water cell for coastal land
        harbor = np.zeros(n_cells, dtype=np.uint8)  # Number of adjacent water cells
        features = [None]  # index 0 is reserved

        # Use DFS to identify connected features (matching FMG's behavior)
        queue = [0]
        feature_id = 1

        while len(queue) > 0 and queue[0] != -1:
            first_cell = queue[0]
            feature_ids[first_cell] = feature_id

            land = self._is_land_packed(packed_graph, first_cell)
            border = bool(packed_graph.cell_border_flags[first_cell])
            total_cells = 1

            # DFS to mark all cells in this feature (using pop() like FMG)
            while queue:
                cell_id = queue.pop()

                # Check if on border
                if packed_graph.cell_border_flags[cell_id]:
                    border = True

                # Check all neighbors
                for neighbor_id in packed_graph.cell_neighbors[cell_id]:
                    is_neib_land = self._is_land_packed(packed_graph, neighbor_id)

                    # Mark coastline and define haven/harbor
                    if land and not is_neib_land:
                        distance_field[cell_id] = LAND_COAST
                        distance_field[neighbor_id] = WATER_COAST
                        if haven[cell_id] == 0:
                            self._define_haven(packed_graph, cell_id, haven, harbor)
                    elif land and is_neib_land:
                        # Handle landlocked cells
                        if (
                            distance_field[neighbor_id] == UNMARKED
                            and distance_field[cell_id] == LAND_COAST
                        ):
                            distance_field[neighbor_id] = LANDLOCKED
                        elif (
                            distance_field[cell_id] == UNMARKED
                            and distance_field[neighbor_id] == LAND_COAST
                        ):
                            distance_field[cell_id] = LANDLOCKED

                    # Add to feature if same type and unmarked
                    if feature_ids[neighbor_id] == 0 and land == is_neib_land:
                        queue.append(neighbor_id)
                        feature_ids[neighbor_id] = feature_id
                        total_cells += 1

            # Create feature with vertices and area calculation
            feature = self._add_packed_feature(
                packed_graph,
                first_cell,
                land,
                border,
                feature_id,
                total_cells,
                feature_ids,
            )
            features.append(feature)

            # Find next unmarked cell
            try:
                next_unmarked = next(
                    i for i, f in enumerate(feature_ids) if f == UNMARKED
                )
                queue.append(next_unmarked)
            except StopIteration:
                queue.append(-1)  # No more unmarked cells

            feature_id += 1

        # Markup deeper land cells (DEEPER_LAND = 3)
        self._markup_distance_field_packed(
            distance_field, packed_graph.cell_neighbors, start=DEEPER_LAND, increment=1
        )

        # Markup deep ocean cells
        self._markup_distance_field_packed(
            distance_field,
            packed_graph.cell_neighbors,
            start=DEEP_WATER,
            increment=-1,
            limit=-10,
        )

        # Store results on packed graph
        packed_graph.distance_field = distance_field
        packed_graph.feature_ids = feature_ids
        packed_graph.haven = haven
        packed_graph.harbor = harbor
        packed_graph.features = features

    def _is_land_packed(self, packed_graph, cell_id: int) -> bool:
        """Check if a packed cell is land based on its height."""
        return packed_graph.heights[cell_id] >= 20

    def _define_haven(
        self, packed_graph, cell_id: int, haven: np.ndarray, harbor: np.ndarray
    ):
        """Define haven (closest water cell) and harbor (water neighbor count)."""
        water_cells = []
        for neighbor_id in packed_graph.cell_neighbors[cell_id]:
            if not self._is_land_packed(packed_graph, neighbor_id):
                water_cells.append(neighbor_id)

        if not water_cells:
            return

        # Find closest water cell
        cell_point = packed_graph.points[cell_id]
        distances = []
        for water_id in water_cells:
            water_point = packed_graph.points[water_id]
            dist = (cell_point[0] - water_point[0]) ** 2 + (
                cell_point[1] - water_point[1]
            ) ** 2
            distances.append(dist)

        closest_idx = np.argmin(distances)
        haven[cell_id] = water_cells[closest_idx]
        harbor[cell_id] = len(water_cells)

    def _add_packed_feature(
        self,
        packed_graph,
        first_cell: int,
        land: bool,
        border: bool,
        feature_id: int,
        total_cells: int,
        feature_ids: np.ndarray,
    ) -> Feature:
        """Create a packed feature with vertices and area calculation."""
        feature_type = "island" if land else ("ocean" if border else "lake")

        # For ocean features, we don't need vertices
        if feature_type == "ocean":
            return Feature(
                id=feature_id,
                type=feature_type,
                land=land,
                border=border,
                cells=total_cells,
                first_cell=first_cell,
                vertices=[],
                area=0.0,
            )

        # Find a cell on the feature border
        start_cell = self._find_border_cell(
            packed_graph, first_cell, feature_ids, feature_id
        )

        # Get feature vertices by tracing the perimeter
        feature_vertices = self._get_feature_vertices(
            packed_graph, start_cell, feature_ids, feature_id
        )

        # Calculate area using vertices
        if feature_vertices:
            vertex_points = [
                packed_graph.vertex_coordinates[v] for v in feature_vertices
            ]
            # Clip to bounds and calculate area
            area = self._calculate_polygon_area(vertex_points)
        else:
            area = 0.0

        # Create feature
        feature = Feature(
            id=feature_id,
            type=feature_type,
            land=land,
            border=border,
            cells=total_cells,
            first_cell=start_cell,
            vertices=feature_vertices,
            area=abs(area),
        )

        # Additional properties for lakes
        if feature_type == "lake":
            # Reverse vertices if needed (lakes should have counter-clockwise winding)
            if area > 0:
                feature.vertices = feature.vertices[::-1]

            # Find shoreline cells (land cells adjacent to this lake)
            shoreline = []
            for v in feature.vertices:
                for cell_id in packed_graph.vertex_cells[v]:
                    if cell_id < len(feature_ids) and self._is_land_packed(
                        packed_graph, cell_id
                    ):
                        if cell_id not in shoreline:
                            shoreline.append(cell_id)
            feature.shoreline = shoreline

            # Lake height calculation would go here
            # feature.height = self._calculate_lake_height(feature)

        return feature

    def _find_border_cell(
        self, packed_graph, first_cell: int, feature_ids: np.ndarray, feature_id: int
    ) -> int:
        """Find a cell on the border of the feature."""
        # Check if first cell is on border
        if packed_graph.cell_border_flags[first_cell]:
            return first_cell

        # Check if first cell has neighbors of different type
        for neighbor_id in packed_graph.cell_neighbors[first_cell]:
            if feature_ids[neighbor_id] != feature_id:
                return first_cell

        # Search all cells of this feature for one on border
        for i in range(len(feature_ids)):
            if feature_ids[i] != feature_id:
                continue

            # Check if on map border
            if packed_graph.cell_border_flags[i]:
                return i

            # Check if has neighbors of different type
            for neighbor_id in packed_graph.cell_neighbors[i]:
                if feature_ids[neighbor_id] != feature_id:
                    return i

        # Fallback to first cell
        return first_cell

    def _get_feature_vertices(
        self, packed_graph, start_cell: int, feature_ids: np.ndarray, feature_id: int
    ) -> List[int]:
        """
        Traces the perimeter of a feature. This is a simplified, robust implementation
        focused on correctly navigating the boundary.
        """
        # 1. Find a valid starting vertex on the boundary.
        starting_vertex = -1
        for v in packed_graph.cell_vertices[start_cell]:
            cells_at_v = packed_graph.vertex_cells[v]
            # A boundary vertex is adjacent to cells of different feature types.
            if any(
                c < len(feature_ids) and feature_ids[c] != feature_id
                for c in cells_at_v
            ):
                starting_vertex = v
                break
        if starting_vertex == -1:
            return []

        # 2. Begin the traversal.
        chain = [starting_vertex]
        previous_v = -1
        current_v = starting_vertex

        # Safety break.
        max_iterations = len(packed_graph.vertex_coordinates) + 10

        for _ in range(max_iterations):
            # 3. Find ALL valid next steps from the current vertex.
            candidates = []
            for neighbor_v in packed_graph.vertex_neighbors[current_v]:
                # A valid next step cannot be the vertex we just came from.
                if neighbor_v == previous_v:
                    continue

                # A valid next step must also lie on the boundary.
                shared_cells = [
                    c
                    for c in packed_graph.vertex_cells[current_v]
                    if c in packed_graph.vertex_cells[neighbor_v]
                ]
                if len(shared_cells) == 2:
                    if (feature_ids[shared_cells[0]] == feature_id) != (
                        feature_ids[shared_cells[1]] == feature_id
                    ):
                        candidates.append(neighbor_v)

            # 4. Decide on the next step.
            if not candidates:
                # No path forward. We've hit a dead end (e.g., map edge). Terminate.
                break

            # In a simple perimeter, there should only be one candidate.
            # If there are more, it's a complex intersection. We will arbitrarily pick the first one.
            # This is a simplification but is more robust than complex angle math.
            next_v = candidates[0]

            # 5. Check for completion.
            if next_v == starting_vertex:
                # We have found our way back to the start. The loop is complete.
                break

            # 6. Advance the traversal.
            chain.append(next_v)
            previous_v = current_v
            current_v = next_v

        return chain

    def _calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """Calculate area of a polygon using the shoelace formula."""
        if len(points) < 3:
            return 0.0

        # Shoelace formula
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return area / 2.0

    def _markup_distance_field_packed(
        self,
        distance_field: np.ndarray,
        neighbors: List[List[int]],
        start: int,
        increment: int,
        limit: int = 127,
    ):
        """Calculate distance field for packed cells."""
        distance = start

        while True:
            marked = 0
            prev_distance = distance - increment

            # Find all cells at previous distance
            for cell_id in range(len(distance_field)):
                if distance_field[cell_id] != prev_distance:
                    continue

                # Mark unmarked neighbors
                for neighbor_id in neighbors[cell_id]:
                    if distance_field[neighbor_id] == UNMARKED:
                        distance_field[neighbor_id] = distance
                        marked += 1

            # Stop if no cells marked or limit reached
            if marked == 0 or distance == limit:
                break

            distance += increment
