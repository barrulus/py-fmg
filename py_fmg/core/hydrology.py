"""
Hydrology and river generation system.

This module handles water flow simulation, depression filling, and river formation
following the original Fantasy Map Generator algorithms.

Process:
1. alterHeights() - Modify heightmap for water flow
2. resolveDepressions() - Fill depressions iteratively
3. drainWater() - Simulate water flow and river formation
4. defineRivers() - Create final river segments with properties
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HydrologyOptions:
    """Hydrology calculation options matching FMG's parameters."""
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
    
    # Controls to improve continuity without lowering counts
    precip_multiplier: float = 1.0  # Multiply precipitation when adding flux
    snap_to_coast_steps: int = 3    # BFS steps to snap river mouths to coast
    
    # Topography guidance
    topo_guided_flow: bool = True   # Use local gradient to guide neighbor choice
    uphill_penalty: float = 50.0    # Cost weight when extending mouths uphill (for coast path)
    lake_evaporation_factor: float = 0.3  # Evaporation ~ area * factor (lower -> more outflow)
    
    # Parity width heuristic
    parity_width: bool = False  # If True, use FMG-like discrete width bins for display


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
    # Meandering geometry
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    path_widths: List[float] = field(default_factory=list)
    polygon: Optional[List[List[float]]] = None


class Hydrology:
    """Handles water flow simulation and river generation."""

    def __init__(
        self,
        graph,
        features,
        climate,
        options: Optional[HydrologyOptions] = None,
    ) -> None:
        """
        Initialize Hydrology with graph, features, and climate data.
        
        Args:
            graph: VoronoiGraph instance with populated heights
            features: Features instance with detected lakes and coastlines
            climate: Climate instance with precipitation data
            options: HydrologyOptions for configuration
        """
        self.graph = graph
        self.features = features
        self.climate = climate
        self.options = options or HydrologyOptions()

        # Initialize hydrology arrays
        self.flux = np.zeros(len(graph.points), dtype=np.float32)  # Water flux (m³/s)
        self.river_ids = np.zeros(len(graph.points), dtype=np.int32)  # River ID per cell
        self.confluences = np.zeros(len(graph.points), dtype=bool)  # River confluence markers

        # River data structures
        self.rivers: Dict[int, RiverData] = {}
        self.next_river_id = 1

        # Working arrays for depression resolution
        self.original_heights = None

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

        # Store original heights and promote to float for sub-cell adjustments
        self.original_heights = self.graph.heights.copy()
        if not isinstance(self.graph.heights, np.ndarray) or self.graph.heights.dtype.kind in ('u','i'):
            # Use float for precise +0.1 / +0.2 adjustments like FMG
            self.graph.heights = self.graph.heights.astype(np.float32)

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

        # Helper function to get height of lake or cell (optimized)
        # Use cached feature-by-id lookup if present
        feature_by_id = {}
        try:
            feats = getattr(self.features, 'features', None)
            if isinstance(feats, list):
                for f in feats:
                    if f is None:
                        continue
                    fid = getattr(f, 'id', None)
                    if fid is not None:
                        feature_by_id[int(fid)] = f
        except Exception:
            feature_by_id = {}

        def height(i: int) -> float:
            # Do not cast to int: we work with float heights during hydrology
            h = float(self.graph.heights[i])
            if h >= self.options.sea_level:
                return h
            # For water cells, if part of a lake with assigned height, return elevated lake height
            try:
                fids = getattr(self.features, 'feature_ids', None)
                if fids is not None and i < len(fids):
                    fid = int(fids[i])
                    if fid > 0:
                        ft = feature_by_id.get(fid)
                        if ft is not None and getattr(ft, 'type', None) == 'lake':
                            lake_h = getattr(ft, 'height', None)
                            if lake_h is not None:
                                return float(lake_h)
                            return h + 0.1
            except Exception:
                pass
            return h

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

        stagnation = 0
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
                if depressions >= prev_depressions:
                    stagnation += 1
                else:
                    stagnation = 0
            prev_depressions = depressions

            # Check if converged
            if depressions == 0:
                logger.info(f"Depression resolution converged after {iteration + 1} iterations")
                break
            # Stop if not making progress for several iterations
            if stagnation >= 5:
                logger.warning("Depression resolution stagnated; aborting early", remaining=depressions)
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
        """Pre-calc lake climate/outlets using FMG-parity logic from Lakes module."""
        # Reuse precomputed mapping if API/CLI prepared it between Climate and Hydrology
        try:
            cached = getattr(self.graph, 'lake_out_cells', None)
            if isinstance(cached, dict):
                return cached
        except Exception:
            pass
        try:
            from .lakes import define_climate_data
            return define_climate_data(self.graph, self.climate)
        except Exception:
            return {}

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

        # Helper to check permafrost
        def is_permafrost(cid: int) -> bool:
            try:
                if hasattr(self.climate, 'temperatures') and self.climate.temperatures is not None:
                    if hasattr(self.graph, 'grid_indices') and self.graph.grid_indices is not None:
                        gid = self.graph.grid_indices[cid]
                    else:
                        gid = cid
                    t = float(self.climate.temperatures[gid]) if gid < len(self.climate.temperatures) else 0.0
                    return t < getattr(self.climate.options, 'permafrost_threshold', -5.0)
            except Exception:
                return False
            return False

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

            # Do not accumulate flux in permafrost (glacial) cells
            if not is_permafrost(cell_id):
                self.flux[cell_id] += (precip * self.options.precip_multiplier) / cells_number_modifier

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
            # Policy: do not START rivers inside permafrost, but allow
            # existing rivers to flow across permafrost toward the sea.
            if is_permafrost(cell_id) and self.river_ids[cell_id] == 0:
                continue
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
        """Select downstream neighbor, guided by local gradient when enabled."""
        neighbors = self._get_neighbors(cell_id)
        if not neighbors:
            return None

        h0 = self.graph.heights[cell_id]

        # Fast path: strictly lower neighbor exists → choose the one best aligned with descent
        lower_neighbors = [n for n in neighbors if self.graph.heights[n] < h0]
        if lower_neighbors:
            if not self.options.topo_guided_flow:
                # Choose steepest drop
                return min(lower_neighbors, key=lambda n: self.graph.heights[n])
            # Compute local descent direction from gradients
            dx, dy = self._local_descent_vector(cell_id)
            # Fallback to steepest if flat
            if dx == 0.0 and dy == 0.0:
                return min(lower_neighbors, key=lambda n: self.graph.heights[n])
            # Pick neighbor with strong alignment and low height
            best = None
            best_score = None
            px, py = self.graph.points[cell_id]
            for n in lower_neighbors:
                qx, qy = self.graph.points[n]
                vx, vy = qx - px, qy - py
                dist = math.hypot(vx, vy) or 1.0
                ux, uy = vx / dist, vy / dist
                align = max(0.0, (ux * dx + uy * dy))  # [0..1]
                drop = max(0.0, float(h0 - self.graph.heights[n]))
                score = drop + 0.5 * align
                if best is None or score > best_score:
                    best = n
                    best_score = score
            return best

        # No lower neighbor: avoid climbing ridges; choose minimal ascent aligned with descent
        best = None
        best_cost = None
        dx, dy = self._local_descent_vector(cell_id)
        px, py = self.graph.points[cell_id]
        for n in neighbors:
            nh = float(self.graph.heights[n])
            rise = max(0.0, nh - float(h0))
            qx, qy = self.graph.points[n]
            vx, vy = qx - px, qy - py
            dist = math.hypot(vx, vy) or 1.0
            ux, uy = vx / dist, vy / dist
            align = max(0.0, (ux * dx + uy * dy))
            # Penalize climbing; prefer moving along descent direction when forced
            cost = rise + (1.0 - align) * 0.05
            if best is None or cost < best_cost:
                best = n
                best_cost = cost
        return best

    def _find_flow_target_excluding_lakes(self, cell_id: int, lake_ids: List[int]) -> Optional[int]:
        """Select downstream neighbor excluding lake cells, with topo guidance."""
        neighbors = self._get_neighbors(cell_id)
        if not neighbors:
            return None

        # Exclude neighbors in given lakes
        filtered = []
        for nid in neighbors:
            in_lake = False
            if (
                hasattr(self.features, 'feature_ids') and self.features.feature_ids is not None and
                nid < len(self.features.feature_ids)
            ):
                fid = int(self.features.feature_ids[nid])
                if fid in lake_ids:
                    in_lake = True
            if not in_lake:
                filtered.append(nid)
        if not filtered:
            return None

        # Temporarily replace neighbor list for guided selection
        original_neighbors = self.graph.cell_neighbors[cell_id]
        self.graph.cell_neighbors[cell_id] = filtered
        try:
            return self._find_flow_target(cell_id)
        finally:
            self.graph.cell_neighbors[cell_id] = original_neighbors

    def _local_descent_vector(self, cell_id: int) -> tuple[float, float]:
        """Estimate unit descent direction from neighbor heights."""
        px, py = self.graph.points[cell_id]
        h0 = float(self.graph.heights[cell_id])
        gx = 0.0
        gy = 0.0
        for n in self._get_neighbors(cell_id):
            qx, qy = self.graph.points[n]
            hn = float(self.graph.heights[n])
            dx = qx - px
            dy = qy - py
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            dh = hn - h0
            ux = dx / dist
            uy = dy / dist
            gx += dh * ux
            gy += dh * uy
        # Descent is negative gradient
        mag = math.hypot(gx, gy)
        if mag == 0:
            return (0.0, 0.0)
        dx = -gx / mag
        dy = -gy / mag
        return (dx, dy)

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
        # Allow rivers to traverse permafrost so mouths can reach the sea.
        # We still refrain from spawning new rivers inside permafrost (handled upstream).
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

            # Build meandered centerline and variable-width polygon path
            try:
                pts, ws = self._build_meandered_path(river)
                river.path_points = pts
                river.path_widths = ws
                ring = self._build_polygon_from_centerline(pts, ws)
                if ring and len(ring) >= 4:
                    # Ensure closed ring
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])
                    river.polygon = ring
            except Exception as e:
                logger.warning("Failed to build meandered path/polygon", river_id=river_id, error=str(e))
        
        # Snap river mouths near coast to the sea with a short BFS if requested
        # Optionally extend mouths toward the nearest ocean
        if self.options.snap_to_coast_steps is not None:
            for river_id, river in list(self.rivers.items()):
                if not river.cells:
                    continue
                mouth = river.cells[-1]
                if self.graph.heights[mouth] < self.options.sea_level:
                    continue  # already at water
                path = self._least_cost_path_to_ocean(mouth, self.options.snap_to_coast_steps, current_river_id=river_id)
                if path:
                    for c in path:
                        if c not in river.cells:
                            river.cells.append(c)
                            self.river_ids[c] = river_id

    def _least_cost_path_to_ocean(self, start: int, max_steps: int | None, current_river_id: Optional[int] = None) -> List[int] | None:
        """Find a near-downhill path from start cell to the ocean using a cost function.

        Strongly penalizes uphill movement and avoids lakes; prefers shorter, downhill paths.
        Returns the path excluding the start cell, or None if not found within limits.
        """
        import heapq

        def is_ocean_cell(cid: int) -> bool:
            try:
                if hasattr(self.features, 'feature_ids') and hasattr(self.graph, 'features') \
                   and self.features.feature_ids is not None and self.graph.features is not None:
                    fid = int(self.features.feature_ids[cid]) if cid < len(self.features.feature_ids) else 0
                    if 0 < fid < len(self.graph.features):
                        f = self.graph.features[fid]
                        if f is not None and getattr(f, 'type', None) == 'ocean':
                            return True
            except Exception:
                pass
            return False

        def is_lake_cell(cid: int) -> bool:
            try:
                if hasattr(self.features, 'feature_ids') and self.features.feature_ids is not None:
                    fid = int(self.features.feature_ids[cid]) if cid < len(self.features.feature_ids) else 0
                    if 0 < fid < len(self.graph.features):
                        f = self.graph.features[fid]
                        return f is not None and getattr(f, 'type', None) == 'lake'
            except Exception:
                pass
            return False

        unlimited = (max_steps is None) or (max_steps <= 0)
        start_h = float(self.graph.heights[start])
        pq = []  # (cost, cell)
        heapq.heappush(pq, (0.0, start))
        dist = {start: 0.0}
        prev: dict[int, int] = {}
        steps = 0
        visited_layers = {start: 0}

        while pq and (unlimited or steps <= max_steps):
            cost, u = heapq.heappop(pq)
            # If reached ocean
            if is_ocean_cell(u):
                # reconstruct path excluding start
                path = []
                cur = u
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path

            # Expand
            for v in self._get_neighbors(u):
                if is_lake_cell(v):
                    continue  # do not route through lakes for mouth snapping
                du = float(self.graph.heights[u])
                dv = float(self.graph.heights[v])
                rise = max(0.0, dv - du)
                step_cost = 1.0 + self.options.uphill_penalty * rise
                ncost = cost + step_cost
                if v not in dist or ncost < dist[v]:
                    dist[v] = ncost
                    prev[v] = u
                    heapq.heappush(pq, (ncost, v))
                    visited_layers[v] = visited_layers[u] + 1
                # Previously we stopped when encountering another river cell to "join" networks.
                # That produced inland termini if the encountered river was not yet extended.
                # We now always continue searching to the ocean.
            steps += 1
        return None

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

        # Parity mode: approximate FMG's visual width scaling using discrete bins
        if getattr(self.options, "parity_width", False):
            # FMG scales stroke width roughly by log/thresholded flux classes.
            # Use simple bins; tune as needed during parity snapshots.
            q = float(discharge)
            if q < 10:
                return 1.0
            if q < 30:
                return 2.0
            if q < 80:
                return 3.0
            if q < 200:
                return 4.0
            if q < 500:
                return 5.0
            return 6.0

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

    # --- Meandering and polygonal path helpers ---
    def _shared_edge_midpoint(self, a: int, b: int) -> List[float]:
        """Midpoint of the shared Voronoi edge between cells a and b.

        Falls back to midpoint of centroids if ridge vertices cannot be found.
        """
        try:
            ca = set(self.graph.cell_vertices[a])
            cb = set(self.graph.cell_vertices[b])
            shared = list(ca.intersection(cb))
            if len(shared) >= 2:
                v1, v2 = shared[0], shared[1]
                p1 = self.graph.vertex_coordinates[v1]
                p2 = self.graph.vertex_coordinates[v2]
                return [float((p1[0] + p2[0]) / 2.0), float((p1[1] + p2[1]) / 2.0)]
        except Exception:
            pass
        # Fallback: midpoint of cell centers
        p0 = self.graph.points[a]
        p1 = self.graph.points[b]
        return [float((p0[0] + p1[0]) / 2.0), float((p0[1] + p1[1]) / 2.0)]

    def _catmull_rom_spline(self, points: List[List[float]], alpha: float = 0.5, segments: int = 8) -> List[List[float]]:
        """Interpolate a Catmull-Rom spline through points with centripetal parameterization."""
        if len(points) < 2:
            return points
        pts = [[float(x), float(y)] for x, y in points]
        # Duplicate endpoints for handling boundaries
        p = [pts[0]] + pts + [pts[-1]]
        out: List[List[float]] = []
        for i in range(1, len(p) - 2):
            p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
            # Compute parameterization
            def tj(ti: float, pa: List[float], pb: List[float]) -> float:
                dx = pb[0] - pa[0]
                dy = pb[1] - pa[1]
                return (dx * dx + dy * dy) ** (alpha * 0.5) + ti
            t0 = 0.0
            t1 = tj(t0, p0, p1)
            t2 = tj(t1, p1, p2)
            t3 = tj(t2, p2, p3)
            for t in np.linspace(t1, t2, max(2, segments), endpoint=True):
                # Interpolate points
                def lerp(pa, pb, ta, tb, t):
                    if tb - ta == 0:
                        return pa
                    w = (t - ta) / (tb - ta)
                    return [pa[0] + (pb[0] - pa[0]) * w, pa[1] + (pb[1] - pa[1]) * w]
                a1 = lerp(p0, p1, t0, t1, t)
                a2 = lerp(p1, p2, t1, t2, t)
                a3 = lerp(p2, p3, t2, t3, t)
                b1 = lerp(a1, a2, t0, t2, t)
                b2 = lerp(a2, a3, t1, t3, t)
                c = lerp(b1, b2, t1, t2, t)
                out.append([float(c[0]), float(c[1])])
        return out

    def _build_meandered_path(self, river: RiverData, alpha: float = 0.5, segments_per_edge: int = 8) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Build a smoothed meandered polyline and per-vertex widths for a river.

        Returns:
            (points, widths) where points is a list of (x,y) and widths is per-point width.
        """
        cells = river.cells
        if not cells or len(cells) < 2:
            return [], []
        waypoints: List[List[float]] = []
        # Start at source cell center
        p0 = self.graph.points[cells[0]]
        waypoints.append([float(p0[0]), float(p0[1])])
        # Add shared edge midpoints along the path
        for i in range(len(cells) - 1):
            a, b = cells[i], cells[i + 1]
            mid = self._shared_edge_midpoint(a, b)
            if not waypoints or mid != waypoints[-1]:
                waypoints.append(mid)
        # If mouth cell is water, extend to center for clarity
        last = cells[-1]
        if int(self.graph.heights[last]) < self.options.sea_level:
            pl = self.graph.points[last]
            last_pt = [float(pl[0]), float(pl[1])]
            if waypoints[-1] != last_pt:
                waypoints.append(last_pt)

        smooth = self._catmull_rom_spline(waypoints, alpha=alpha, segments=segments_per_edge)
        if len(smooth) < 2:
            smooth = waypoints

        # Compute per-point widths increasing toward mouth
        # Width at mouth is precomputed river.width; source width is smaller
        w_mouth = max(1.0, float(river.width))
        w_source = max(0.3 * w_mouth, 1.0)
        n = len(smooth)
        widths: List[float] = []
        for i in range(n):
            t = i / max(1, n - 1)
            widths.append(w_source + (w_mouth - w_source) * t)
        return [(float(x), float(y)) for x, y in smooth], widths

    def _build_polygon_from_centerline(self, points: List[Tuple[float, float]], widths: List[float]) -> List[List[float]]:
        """Construct a variable-width river polygon from centerline points and widths."""
        if not points or len(points) < 2 or len(points) != len(widths):
            return []
        left: List[List[float]] = []
        right: List[List[float]] = []
        n = len(points)
        for i in range(n):
            x, y = points[i]
            # Tangent: forward/back difference
            if i == 0:
                dx = points[i + 1][0] - x
                dy = points[i + 1][1] - y
            elif i == n - 1:
                dx = x - points[i - 1][0]
                dy = y - points[i - 1][1]
            else:
                dx = points[i + 1][0] - points[i - 1][0]
                dy = points[i + 1][1] - points[i - 1][1]
            mag = math.hypot(dx, dy) or 1.0
            tx, ty = dx / mag, dy / mag
            # Normal to the left
            nx, ny = -ty, tx
            hw = widths[i] / 2.0
            left.append([float(x + nx * hw), float(y + ny * hw)])
            right.append([float(x - nx * hw), float(y - ny * hw)])
        ring = left + right[::-1]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring
