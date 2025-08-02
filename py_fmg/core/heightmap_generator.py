"""
Heightmap generation module for fantasy map generation.

This module ports the FMG heightmap generation algorithms to Python,
using NumPy for vectorized operations and better performance.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from ..utils.random import set_random_seed, get_prng
from .voronoi_graph import (
    VoronoiGraph,
    build_cell_connectivity,
    build_cell_vertices,
    build_vertex_connectivity,
)

@dataclass
class HeightmapConfig:
    """Configuration for heightmap generation."""

    width: int
    height: int
    cells_x: int
    cells_y: int
    cells_desired: int = 10000
    spacing: float = 1.0


class HeightmapGenerator:
    """
    Generates heightmaps using various procedural algorithms.

    This is a Python port of FMG's heightmap-generator.js module,
    optimized using NumPy for better performance.
    """

    def __init__(self, config: HeightmapConfig, graph:VoronoiGraph, seed: Optional[str] = None):
        """
        Initialize the heightmap generator.

        Args:
            config: Heightmap configuration
            graph: Voronoi graph structure with cells and connectivity
            seed: Optional seed for PRNG reseeding
        """
        self.config = config
        self.graph = graph
        self.n_cells = len(graph.points)

        # Initialize heights array - use float32 to avoid overflow issues
        # FMG uses regular JavaScript numbers which can handle values > 255
        self.heights = np.zeros(self.n_cells, dtype=np.float32)

        # Calculate power factors based on cell count
        self.blob_power = self._get_blob_power(config.cells_desired)
        self.line_power = self._get_line_power(config.cells_desired)

        # Reseed PRNG if seed provided (matches FMG's heightmap-generator.js:68)
        if seed:
            set_random_seed(seed)
            self._prng = get_prng()
        else:
            self._prng = None

    def _get_blob_power(self, cells: int) -> float:
        """Get blob spreading power factor based on cell count."""
        blob_power_map = {
            1000: 0.93,
            2000: 0.95,
            5000: 0.97,
            10000: 0.98,
            20000: 0.99,
            30000: 0.991,
            40000: 0.993,
            50000: 0.994,
            60000: 0.995,
            70000: 0.9955,
            80000: 0.996,
            90000: 0.9964,
            100000: 0.9973,
        }

        # Match FMG behavior - exact match or default
        return blob_power_map.get(cells, 0.98)

    def _get_line_power(self, cells: int) -> float:
        """Get line spreading power factor based on cell count."""
        line_power_map = {
            1000: 0.75,
            2000: 0.77,
            5000: 0.79,
            10000: 0.81,
            20000: 0.82,
            30000: 0.83,
            40000: 0.84,
            50000: 0.86,
            60000: 0.87,
            70000: 0.88,
            80000: 0.91,
            90000: 0.92,
            100000: 0.93,
        }
        # Replicate JS behavior: exact key match or default
        return line_power_map.get(cells, 0.81)

    def _lim(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Limit values to 0-100 range."""
        # Keep as float to avoid overflow, FMG rounds at display time
        return np.clip(value, 0, 100)

    def _random(self) -> float:
        """Get next random value from Alea PRNG."""
        if self._prng is None:
            self._prng = get_prng()
        return self._prng.random()

    def _rand(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> int:
        """Match FMG's rand() function - returns integer in range [min, max] inclusive."""
        if min_val is None and max_val is None:
            return int(self._random() * 2**32)  # Return large random int
        if max_val is None:
            max_val = min_val
            min_val = 0
        return int(self._random() * (max_val - min_val + 1)) + int(min_val)

    def _P(self, probability: float) -> bool:
        """Match FMG's P() probability function."""
        if probability >= 1:
            return True
        if probability <= 0:
            return False
        return self._random() < probability

    def _get_number_in_range(self, value: Union[int, float, str]) -> float:
        """Parse number range string and return a random value within it."""
        # Convert to string to handle consistently
        r = str(value)

        # Check if it's a simple number (not a range)
        try:
            num = float(r)
            # Match FMG: return ~~r + +P(r - ~~r)
            # ~~ is floor, so integer part + probability of fractional part
            integer_part = int(num)
            fractional_part = num - integer_part
            if fractional_part > 0 and self._P(fractional_part):
                return float(integer_part + 1)
            return float(integer_part)
        except ValueError:
            pass

        # Handle ranges like "5-10"
        if "-" in r:
            # Handle negative sign at start
            sign = 1
            if r[0] == "-":
                sign = -1
                r = r[1:]

            if "-" in r:
                parts = r.split("-")
                min_val = float(parts[0]) * sign
                max_val = float(parts[1])
                return float(self._rand(min_val, max_val))

        # Fallback
        return 0.0

    def _get_point_in_range(self, range_str: str, max_val: float) -> float:
        """Get a random point within the specified range."""
        if "-" in range_str:
            parts = range_str.split("-")
            min_pct = float(parts[0]) / 100 if parts[0] else 0
            max_pct = float(parts[1]) / 100 if parts[1] else min_pct
            # Use _rand to match FMG's getPointInRange exactly
            return float(self._rand(min_pct * max_val, max_pct * max_val))

        pct = float(range_str) / 100
        return max_val * pct

    def _parse_range_bounds(
        self, range_str: str, max_val: float
    ) -> Tuple[float, float]:
        """
        Parse range string to get min and max bounds.

        Args:
            range_str: Range string like "5-95" (percentages)
            max_val: Maximum coordinate value

        Returns:
            Tuple of (min_bound, max_bound) in coordinate units
        """
        if "-" in range_str:
            min_pct, max_pct = map(float, range_str.split("-"))
        else:
            min_pct = max_pct = float(range_str)

        return (min_pct * max_val / 100.0, max_pct * max_val / 100.0)

    def _find_grid_cell(self, x: float, y: float) -> int:
        """Find the grid cell index for a given x,y coordinate."""
        col = min(int(x / self.config.spacing), self.config.cells_x - 1)
        row = min(int(y / self.config.spacing), self.config.cells_y - 1)
        return row * self.config.cells_x + col

    def add_hill(
        self,
        count: Union[int, str],
        height: Union[int, str],
        range_x: str,
        range_y: str,
    ) -> None:
        """
        Add hills to the heightmap using blob spreading algorithm.

        Args:
            count: Number of hills to add
            height: Height range for hills
            range_x: X-coordinate range (percentage)
            range_y: Y-coordinate range (percentage)
        """
        count = int(self._get_number_in_range(count))

        for _ in range(count):
            self._add_one_hill(height, range_x, range_y)

    def _add_one_hill(
        self, height: Union[int, str], range_x: str, range_y: str
    ) -> None:
        """
        Add a single hill using blob spreading.
        This version corrects the NameError and implements the FMG queueing logic.
        """
        # 1. INITIAL SETUP
        # CRITICAL: Use uint8 to match FMG's Uint8Array behavior
        # The integer truncation is essential to the algorithm!
        change = np.zeros(self.n_cells, dtype=np.uint8)
        h = self._lim(self._get_number_in_range(height))

        # 2. FIND STARTING POINT
        limit = 0
        start = -1  # Initialize `start` to a known invalid value BEFORE the loop
        while limit < 50:
            x = self._get_point_in_range(range_x, self.config.width)
            y = self._get_point_in_range(range_y, self.config.height)

            # Define `start` inside the loop for each attempt
            current_attempt_start = self._find_grid_cell(x, y)

            if self.heights[current_attempt_start] + h <= 90:
                start = current_attempt_start  # Assign to the outer `start` variable
                break  # Exit the loop successfully

            limit += 1

        # 3. ROBUSTNESS CHECK: If the loop finished without finding a start, exit.
        if start == -1:
            # This print statement is useful for debugging template issues.
            # print("WARNING: Could not find a valid start point for hill. Skipping.")
            return

        # 4. INITIALIZE BFS
        change[start] = int(h)
        queue = [start]

        # 5. CORE BFS LOOP
        while queue:
            current = queue.pop(0)
            val_from_array = change[current]

            for neighbor in self.graph.cell_neighbors[current]:
                if change[neighbor] > 0:
                    continue

                # Calculate the float value
                new_height_float = (val_from_array**self.blob_power) * (
                    self._random() * 0.2 + 0.9
                )

                # CRITICAL: Store the value (will be truncated to int by uint8 array)
                change[neighbor] = new_height_float

                # Check the STORED truncated value, not the float!
                if change[neighbor] > 1:
                    queue.append(neighbor)

        # 6. APPLY CHANGES
        self.heights = self._lim(self.heights + change)

    def add_pit(
        self,
        count: Union[int, str],
        height: Union[int, str],
        range_x: str,
        range_y: str,
    ) -> None:
        """
        Add pits (depressions) to the heightmap.

        Args:
            count: Number of pits to add
            height: Depth range for pits
            range_x: X-coordinate range (percentage)
            range_y: Y-coordinate range (percentage)
        """
        count = int(self._get_number_in_range(count))

        for _ in range(count):
            self._add_one_pit(height, range_x, range_y)

    def _add_one_pit(self, height: Union[int, str], range_x: str, range_y: str) -> None:
        """Add a single pit using blob spreading."""
        used = np.zeros(self.n_cells, dtype=bool)
        h = self._get_number_in_range(height)  # Don't limit initial h value

        # Find starting point (prefer non-water cells)
        limit = 0
        while limit < 50:
            x = self._get_point_in_range(range_x, self.config.width)
            y = self._get_point_in_range(range_y, self.config.height)
            start = self._find_grid_cell(x, y)

            if self.heights[start] >= 20:  # Not water
                break
            limit += 1

        # Spread depth using BFS
        queue = [start]
        used[start] = True  # Mark start as used

        while queue:
            current = queue.pop(0)
            h = (h**self.blob_power) * (self._random() * 0.2 + 0.9)

            if h < 1:
                break

            for neighbor in self.graph.cell_neighbors[current]:
                # CRITICAL FIX: Check if used and SKIP (continue) if already processed
                if used[neighbor]:
                    continue  # Skip cells that have already been processed

                # Process this cell ONCE
                depth_factor = h * (self._random() * 0.2 + 0.9)
                self.heights[neighbor] = self._lim(
                    self.heights[neighbor] - depth_factor
                )
                used[neighbor] = True
                queue.append(neighbor)

    def add_range(
        self,
        count: Union[int, str],
        height: Union[int, str],
        range_x: Optional[str] = None,
        range_y: Optional[str] = None,
        start_cell: Optional[int] = None,
        end_cell: Optional[int] = None,
    ) -> None:
        """
        Add mountain ranges to the heightmap.

        Args:
            count: Number of ranges to add
            height: Height range for mountains
            range_x: X-coordinate range for start point
            range_y: Y-coordinate range for start point
            start_cell: Optional specific start cell
            end_cell: Optional specific end cell
        """
        count = int(self._get_number_in_range(count))

        for _ in range(count):
            self._add_one_range(height, range_x, range_y, start_cell, end_cell)

    def _add_one_range(
        self,
        height: Union[int, str],
        range_x: Optional[str],
        range_y: Optional[str],
        start_cell: Optional[int],
        end_cell: Optional[int],
    ) -> None:
        """Add a single mountain range."""
        used = np.zeros(self.n_cells, dtype=bool)
        h = self._lim(self._get_number_in_range(height))

        # Determine start and end cells
        if range_x and range_y and (start_cell is None or end_cell is None):
            # Find start point
            start_x = self._get_point_in_range(range_x, self.config.width)
            start_y = self._get_point_in_range(range_y, self.config.height)
            start_cell = self._find_grid_cell(start_x, start_y)

            # Find end point with appropriate distance
            limit = 0
            end_x = None
            end_y = None
            while limit < 50:
                end_x = (
                    self._random() * self.config.width * 0.8 + self.config.width * 0.1
                )
                end_y = (
                    self._random() * self.config.height * 0.7
                    + self.config.height * 0.15
                )
                dist = abs(end_y - start_y) + abs(end_x - start_x)

                if self.config.width / 8 <= dist <= self.config.width / 3:
                    break
                limit += 1

            # Always set end_cell using the last generated coordinates
            end_cell = self._find_grid_cell(end_x, end_y)

        # Get ridge path
        ridge = self._get_range_path(start_cell, end_cell, used)

        # Add height to ridge and surrounding cells
        queue = list(ridge)
        iteration = 0

        while queue and h >= 2:
            frontier = queue[:]
            queue = []
            iteration += 1

            # Add height to frontier cells
            for cell in frontier:
                height_add = h * (self._random() * 0.3 + 0.85)
                self.heights[cell] = self._lim(self.heights[cell] + height_add)

            # Decay height
            h = h**self.line_power - 1

            # Add neighbors to queue
            for cell in frontier:
                for neighbor in self.graph.cell_neighbors[cell]:
                    if not used[neighbor]:
                        queue.append(neighbor)
                        used[neighbor] = True

        # Generate prominences along ridge
        for i, cell in enumerate(ridge):
            if i % 6 == 0:  # Every 6th cell
                self._add_prominence(cell, iteration)

    def _get_range_path(
        self, start: Optional[int], end: Optional[int], used: np.ndarray
    ) -> List[int]:
        """Find a path from start to end cell for mountain range."""
        # Validate indices
        if start is None:
            raise ValueError("Start cell is None")
        if end is None:
            raise ValueError("End cell is None")
        if start < 0 or start >= len(self.graph.points):
            raise ValueError(
                f"Invalid start index: {start} (total points: {len(self.graph.points)})"
            )
        if end < 0 or end >= len(self.graph.points):
            raise ValueError(
                f"Invalid end index: {end} (total points: {len(self.graph.points)})"
            )

        path = [start]
        current = start
        used[current] = True

        while current != end:
            min_dist = float("inf")
            next_cell = None

            # Find neighbor closest to end
            for neighbor in self.graph.cell_neighbors[current]:
                if used[neighbor]:
                    continue

                # Calculate distance to end
                diff_x = self.graph.points[end][0] - self.graph.points[neighbor][0]
                diff_y = self.graph.points[end][1] - self.graph.points[neighbor][1]
                dist = diff_x**2 + diff_y**2

                # Add randomness to path
                if self._random() > 0.85:
                    dist = dist / 2

                if dist < min_dist:
                    min_dist = dist
                    next_cell = neighbor

            if next_cell is None:
                break

            path.append(next_cell)
            used[next_cell] = True
            current = next_cell

        return path

    def _add_prominence(self, start_cell: int, iterations: int) -> None:
        """Add prominence (peak) extending from a ridge cell."""
        current = start_cell

        for _ in range(iterations):
            # Find lowest neighbor (downhill)
            neighbors = self.graph.cell_neighbors[current]
            if not neighbors:
                break

            # Match FMG's d3.scan usage - find index of minimum in neighbors array
            min_idx = np.argmin([self.heights[n] for n in neighbors])
            min_cell = neighbors[min_idx]

            # Interpolate height
            self.heights[min_cell] = (
                self.heights[current] * 2 + self.heights[min_cell]
            ) / 3
            current = min_cell

    def add_trough(
        self,
        count: Union[int, str],
        height: Union[int, str],
        range_x: Optional[str] = None,
        range_y: Optional[str] = None,
        start_cell: Optional[int] = None,
        end_cell: Optional[int] = None,
    ) -> None:
        """
        Add valleys (troughs) to the heightmap.

        Args:
            count: Number of troughs to add
            height: Depth range for valleys
            range_x: X-coordinate range for start point
            range_y: Y-coordinate range for start point
            start_cell: Optional specific start cell
            end_cell: Optional specific end cell
        """
        count = int(self._get_number_in_range(count))

        for _ in range(count):
            self._add_one_trough(height, range_x, range_y, start_cell, end_cell)

    def _add_one_trough(
        self,
        height: Union[int, str],
        range_x: Optional[str],
        range_y: Optional[str],
        start_cell: Optional[int],
        end_cell: Optional[int],
    ) -> None:
        """Add a single valley."""
        used = np.zeros(self.n_cells, dtype=bool)
        h = self._lim(self._get_number_in_range(height))

        # Determine start and end cells
        if range_x and range_y and (start_cell is None or end_cell is None):
            # Find start point (prefer non-water)
            limit = 0
            while limit < 50:
                start_x = self._get_point_in_range(range_x, self.config.width)
                start_y = self._get_point_in_range(range_y, self.config.height)
                start_cell = self._find_grid_cell(start_x, start_y)

                if self.heights[start_cell] >= 20:
                    break
                limit += 1

            # Find end point
            limit = 0
            end_x = None
            end_y = None
            while limit < 50:
                end_x = (
                    self._random() * self.config.width * 0.8 + self.config.width * 0.1
                )
                end_y = (
                    self._random() * self.config.height * 0.7
                    + self.config.height * 0.15
                )
                dist = abs(end_y - start_y) + abs(end_x - start_x)

                if self.config.width / 8 <= dist <= self.config.width / 2:
                    break
                limit += 1

            # Always set end_cell using the last generated coordinates
            end_cell = self._find_grid_cell(end_x, end_y)

        # Get valley path
        valley = self._get_range_path(start_cell, end_cell, used)

        # Carve valley
        queue = list(valley)
        iteration = 0

        while queue and h >= 2:
            frontier = queue[:]
            queue = []
            iteration += 1

            # Remove height from frontier cells
            for cell in frontier:
                depth = h * (self._random() * 0.3 + 0.85)
                self.heights[cell] = self._lim(self.heights[cell] - depth)

            # Decay depth
            h = h**self.line_power - 1

            # Add neighbors to queue
            for cell in frontier:
                for neighbor in self.graph.cell_neighbors[cell]:
                    if not used[neighbor]:
                        queue.append(neighbor)
                        used[neighbor] = True

    def add_strait(
        self, width: Union[int, str], direction: str = "vertical", *unused
    ) -> None:
        """
        Add a strait (water channel) to the heightmap.

        Args:
            width: Width of the strait
            direction: "vertical" or "horizontal"
        """
        width_raw = self._get_number_in_range(width)
        width_int = min(int(width_raw), self.config.cells_x // 3)

        # Match FMG: if (width < 1 && P(width)) return;
        if width_int < 1:
            if self._P(width_raw):
                return
            # If P(width) fails, continue with width=0 (which will exit the loop immediately)

        used = np.zeros(self.n_cells, dtype=bool)
        is_vertical = direction == "vertical"

        # Determine start and end points
        if is_vertical:
            start_x = self._random() * self.config.width * 0.4 + self.config.width * 0.3
            start_y = 5
            end_x = (
                self.config.width
                - start_x
                - self.config.width * 0.1
                + self._random() * self.config.width * 0.2
            )
            end_y = self.config.height - 5
        else:
            start_x = 5
            start_y = (
                self._random() * self.config.height * 0.4 + self.config.height * 0.3
            )
            end_x = self.config.width - 5
            end_y = (
                self.config.height
                - start_y
                - self.config.height * 0.1
                + self._random() * self.config.height * 0.2
            )

        start_cell = self._find_grid_cell(start_x, start_y)
        end_cell = self._find_grid_cell(end_x, end_y)

        # Get strait path
        path = self._get_strait_path(start_cell, end_cell)

        # Carve strait with decreasing effect - match FMG's simpler logic
        step = 0.1 / width_int

        for w in range(width_int, 0, -1):
            exp = 0.9 - step * w
            next_layer = []

            for cell in path:
                for neighbor in self.graph.cell_neighbors[cell]:
                    if not used[neighbor]:
                        used[neighbor] = True
                        next_layer.append(neighbor)
                        # Apply exponential lowering like FMG
                        self.heights[neighbor] = self.heights[neighbor] ** exp
                        # FMG's strange edge case for values over 100
                        if self.heights[neighbor] > 100:
                            self.heights[neighbor] = 5

            path = next_layer

    def _get_strait_path(self, start: int, end: int) -> List[int]:
        """Find a path for strait without marking cells as used."""
        path = []
        current = start

        while current != end:
            min_dist = float("inf")
            next_cell = None

            # Find neighbor closest to end
            for neighbor in self.graph.cell_neighbors[current]:
                diff_x = self.graph.points[end][0] - self.graph.points[neighbor][0]
                diff_y = self.graph.points[end][1] - self.graph.points[neighbor][1]
                dist = diff_x**2 + diff_y**2

                if self._random() > 0.8:
                    dist = dist / 2

                if dist < min_dist:
                    min_dist = dist
                    next_cell = neighbor

            if next_cell is None:
                break

            path.append(next_cell)
            current = next_cell

        return path

    def smooth(
        self, factor: Union[int, str] = 2, add: Union[float, str] = 0, *unused
    ) -> None:
        """
        Smooth the heightmap by averaging with neighbors.

        Args:
            factor: Smoothing factor (higher = less smoothing)
            add: Value to add after smoothing
            *unused: Ignore extra arguments from template
        """
        factor = self._get_number_in_range(factor) if factor else 2
        add = self._get_number_in_range(add) if add else 0
        new_heights = np.zeros_like(self.heights, dtype=np.float32)

        for i in range(self.n_cells):
            # Get heights of cell and neighbors
            heights_list = [self.heights[i]]
            for neighbor in self.graph.cell_neighbors[i]:
                heights_list.append(self.heights[neighbor])

            # Calculate smoothed height
            if factor == 1:
                new_heights[i] = np.mean(heights_list) + add
            else:
                avg = np.mean(heights_list)
                new_heights[i] = (self.heights[i] * (factor - 1) + avg + add) / factor

        self.heights = self._lim(new_heights)

    def mask(self, power: Union[float, str] = 1, *unused) -> None:
        """
        Apply radial mask to heightmap (fade edges).

        Args:
            power: Mask strength (negative inverts)
            *unused: Ignore extra arguments from template
        """
        power = (
            self._get_number_in_range(power)
            if isinstance(power, str)
            else float(power) if power else 1
        )
        factor = abs(power)
        new_heights = np.zeros_like(self.heights, dtype=np.float32)

        for i in range(self.n_cells):
            x, y = self.graph.points[i]
            # Normalize to [-1, 1] range
            nx = 2 * x / self.config.width - 1
            ny = 2 * y / self.config.height - 1

            # Calculate distance from center (1 at center, 0 at edge)
            distance = (1 - nx**2) * (1 - ny**2)

            if power < 0:
                distance = 1 - distance  # Invert

            # Apply mask
            masked = self.heights[i] * distance
            new_heights[i] = (self.heights[i] * (factor - 1) + masked) / factor

        self.heights = self._lim(new_heights)

    def modify(
        self,
        range_spec: str,
        add: float = 0,
        multiply: float = 1,
        power: Optional[float] = None,
    ) -> None:
        """
        Modify heights within specified range.

        Args:
            range_spec: "all", "land", or "min-max"
            add: Value to add
            multiply: Value to multiply by
            power: Optional power to raise to
        """
        # Parse range
        if range_spec == "land":
            min_h, max_h = 20, 100
            is_land = True
        elif range_spec == "all":
            min_h, max_h = 0, 100
            is_land = False
        else:
            parts = range_spec.split("-")
            min_h = float(parts[0])
            max_h = float(parts[1])
            is_land = min_h == 20

        # Apply modifications
        mask = (self.heights >= min_h) & (self.heights <= max_h)

        if add != 0:
            if is_land:
                self.heights[mask] = np.maximum(self.heights[mask] + add, 20)
            else:
                self.heights[mask] = self.heights[mask] + add

        if multiply != 1:
            if is_land:
                self.heights[mask] = (self.heights[mask] - 20) * multiply + 20
            else:
                self.heights[mask] = self.heights[mask] * multiply

        if power is not None:
            if is_land:
                self.heights[mask] = (self.heights[mask] - 20) ** power + 20
            else:
                self.heights[mask] = self.heights[mask] ** power

        self.heights = self._lim(self.heights)

    def invert(
        self, probability: Union[float, str], axes: str = "both", *unused
    ) -> None:
        """
        Invert (flip) the heightmap.

        Args:
            probability: Probability of inversion (0-1)
            axes: "x", "y", or "both"
        """
        prob = self._get_number_in_range(probability)
        # Match FMG: if (!P(count)) return;
        if not self._P(prob):
            return

        invert_x = axes != "y"
        invert_y = axes != "x"

        # Create new heights array
        new_heights = np.zeros_like(self.heights)

        for i in range(self.n_cells):
            x = i % self.config.cells_x
            y = i // self.config.cells_x

            # Calculate inverted position
            nx = (self.config.cells_x - x - 1) if invert_x else x
            ny = (self.config.cells_y - y - 1) if invert_y else y

            # Calculate inverted index
            inverted_idx = ny * self.config.cells_x + nx

            # Read from inverted position, write to current position
            if 0 <= inverted_idx < self.n_cells:
                new_heights[i] = self.heights[inverted_idx]

        self.heights = new_heights

    def from_template(
        self, template_name: str, seed: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate heightmap from a template name.

        Args:
            template_name: Name of template to load
            seed: Optional random seed

        Returns:
            Generated heights array
        """
        if seed:
            # Reseed PRNG for heightmap generation (matches FMG's Math.random = aleaPRNG(seed))
            set_random_seed(seed)
            # Reset our cached PRNG reference to use the new seed
            self._prng = get_prng()

        # Load template by name
        from ..config.heightmap_templates import get_template

        template = get_template(template_name)

        # Parse and execute template commands
        lines = template.strip().split("\n")

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            command = parts[0]
            args = parts[1:]

            if command == "Hill":
                self.add_hill(*args)
            elif command == "Pit":
                self.add_pit(*args)
            elif command == "Range":
                self.add_range(*args)
            elif command == "Trough":
                self.add_trough(*args)
            elif command == "Strait":
                self.add_strait(*args)
            elif command == "Smooth":
                self.smooth(*args)
            elif command == "Mask":
                self.mask(float(args[0]))
            elif command == "Add":
                # Template format: Add value range unused unused
                self.modify(args[1], add=float(args[0]))
            elif command == "Multiply":
                # Template format: Multiply value range unused unused
                self.modify(args[1], multiply=float(args[0]))
            elif command == "Invert":
                self.invert(*args)

        # Simulate JavaScript's Uint8Array truncation behavior
        # When JS assigns float array to Uint8Array, it truncates (floors) values
        return np.floor(self.heights).astype(np.uint8)
