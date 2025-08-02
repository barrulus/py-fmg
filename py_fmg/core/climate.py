"""
Climate calculation system for temperature and precipitation.

This module implements:
- Latitude-based temperature bands
- Altitude temperature drop
- Wind patterns and precipitation simulation
- Orographic effects and rain shadows
"""

import numpy as np
import structlog
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = structlog.get_logger()


@dataclass
class ClimateOptions:
    """Climate calculation options matching FMG's parameters."""

    # Temperature settings
    temperature_equator: float = 25.0  # °C at equator
    temperature_north_pole: float = -30.0  # °C at north pole
    temperature_south_pole: float = -30.0  # °C at south pole
    height_exponent: float = 1.5  # Exponent for altitude calculations
    temperature_lapse_rate: float = 6.5  # °C per km altitude drop

    # Tropical zone boundaries
    tropic_north: int = 16  # Northern tropic latitude
    tropic_south: int = -20  # Southern tropic latitude
    tropical_gradient: float = 0.15  # Temperature gradient in tropics

    # Precipitation settings
    precipitation_modifier: float = 1.0  # Global precipitation modifier
    base_precipitation_west: float = 120.0  # Base westerly precipitation
    base_precipitation_vertical: float = 60.0  # Base vertical wind precipitation
    water_humidity_gain: float = 5.0  # Humidity gain over water
    water_precipitation: float = 5.0  # Base precipitation over water

    # Wind system
    winds: List[int] = None  # Wind angles by latitude tier
    wind_tier_span: int = 30  # Degrees per wind tier

    # Orographic effects
    max_passable_elevation: int = 85  # Maximum elevation winds can cross
    terrain_mod_threshold: float = 70.0  # Height threshold for terrain modifier
    precipitation_base_divisor: float = 10.0  # Base divisor for precipitation loss
    # Random range for coastal precipitation
    coastal_precip_range: Tuple[int, int] = (10, 21)

    # Evaporation and humidity
    evaporation_threshold: float = 1.5  # Precipitation threshold for evaporation
    permafrost_threshold: float = -5.0  # Temperature threshold for permafrost

    # Latitude precipitation modifiers (by 5-degree bands)
    latitude_precipitation_modifiers: List[float] = None

    def __post_init__(self):
        if self.winds is None:
            # Default wind angles by tier (0-5 from N to S)
            # Based on prevailing winds: polar easterlies, westerlies, trade winds
            self.winds = [
                225,  # Tier 0: Polar easterlies (NE to SW)
                135,  # Tier 1: Westerlies (SE to NW)
                225,  # Tier 2: Trade winds (NE to SW)
                225,  # Tier 3: Trade winds (NE to SW)
                135,  # Tier 4: Westerlies (SE to NW)
                225,  # Tier 5: Polar easterlies (NE to SW)
            ]

        if self.latitude_precipitation_modifiers is None:
            # Precipitation modifiers based on atmospheric circulation cells
            # (Hadley, Ferrel, Polar cells affect precipitation patterns)
            self.latitude_precipitation_modifiers = [
                4.0,  # 0-5°: Wet all year (ITCZ)
                2.0,
                2.0,  # 5-20°: Wet summer, dry winter
                1.0,
                1.0,  # 20-30°: Dry all year (descending air)
                2.0,
                2.0,  # 30-40°: Wet winter, dry summer
                2.0,
                2.0,  # 40-50°: Wet winter, dry summer
                3.0,
                3.0,  # 50-60°: Wet all year
                2.0,
                2.0,  # 60-70°: Wet summer, dry winter
                1.0,
                1.0,  # 70-80°: Dry all year
                1.0,  # 80-85°: Dry all year
                0.5,  # 85-90°: Very dry
            ]


@dataclass
class MapCoordinates:
    """Map latitude boundaries."""

    lat_n: float = 90  # Northern latitude boundary
    lat_s: float = -90  # Southern latitude boundary

    @property
    def lat_t(self):
        """Total latitude span."""
        return self.lat_n - self.lat_s


class Climate:
    """Handles temperature and precipitation calculations."""

    def __init__(
        self,
        graph,
        options: Optional[ClimateOptions] = None,
        map_coords: Optional[MapCoordinates] = None,
    ):
        """
        Initialize climate calculator.

        Args:
            graph: VoronoiGraph with heights populated
            options: Climate calculation options
            map_coords: Map coordinate boundaries
        """
        self.graph = graph
        self.options = options or ClimateOptions()
        self.map_coords = map_coords or MapCoordinates()

        # Temperature and precipitation arrays
        self.temperatures = None
        self.precipitation = None

    def calculate_temperatures(self):
        """
        Calculate temperature for each cell based on latitude and altitude.

        Port of FMG's calculateTemperatures() from main.js:897-943
        """
        logger.info("Calculating temperatures")

        n_cells = len(self.graph.points)
        self.temperatures = np.zeros(n_cells, dtype=np.int8)

        # Tropical zone boundaries from options
        tropics = [self.options.tropic_north, self.options.tropic_south]
        tropical_gradient = self.options.tropical_gradient

        # Calculate temperature at tropic boundaries
        temp_north_tropic = (
            self.options.temperature_equator - tropics[0] * tropical_gradient
        )
        temp_south_tropic = (
            self.options.temperature_equator + tropics[1] * tropical_gradient
        )

        # Calculate gradients for temperate zones
        northern_gradient = (
            temp_north_tropic - self.options.temperature_north_pole
        ) / (90 - tropics[0])
        southern_gradient = (
            temp_south_tropic - self.options.temperature_south_pole
        ) / (90 + tropics[1])

        # Process cells row by row for efficiency
        cells_x = self.graph.cells_x
        graph_height = self.graph.graph_height

        for row_start in range(0, n_cells, cells_x):
            # Calculate latitude for this row
            y = self.graph.points[row_start][1]
            row_latitude = (
                self.map_coords.lat_n - (y / graph_height) * self.map_coords.lat_t
            )

            # Sea level temperature for this latitude
            temp_sea_level = self._calculate_sea_level_temp(
                row_latitude,
                tropics,
                tropical_gradient,
                temp_north_tropic,
                temp_south_tropic,
                northern_gradient,
                southern_gradient,
            )

            # Apply to all cells in row
            for cell_id in range(row_start, min(row_start + cells_x, n_cells)):
                # Calculate altitude temperature drop
                altitude_drop = self._get_altitude_temperature_drop(
                    self.graph.heights[cell_id]
                )

                # Final temperature (clamped to int8 range)
                temp = temp_sea_level - altitude_drop
                self.temperatures[cell_id] = int(np.clip(temp, -128, 127))

        # Store on graph
        self.graph.temperatures = self.temperatures

    def _calculate_sea_level_temp(
        self,
        latitude: float,
        tropics: List[int],
        tropical_gradient: float,
        temp_north_tropic: float,
        temp_south_tropic: float,
        northern_gradient: float,
        southern_gradient: float,
    ) -> float:
        """Calculate sea level temperature for given latitude."""
        # Check if in tropical zone
        is_tropical = latitude <= tropics[0] and latitude >= tropics[1]

        if is_tropical:
            return self.options.temperature_equator - abs(latitude) * tropical_gradient

        # Temperate zones
        if latitude > 0:
            return temp_north_tropic - (latitude - tropics[0]) * northern_gradient
        else:
            return temp_south_tropic + (latitude - tropics[1]) * southern_gradient

    def _get_altitude_temperature_drop(self, height: int) -> float:
        """
        Calculate temperature drop due to altitude.
        Uses configurable lapse rate (default 6.5°C per 1km).
        """
        if height < 20:  # Sea level
            return 0

        # Convert height to km using exponent
        height_km = ((height - 18) ** self.options.height_exponent) / 1000
        return round(height_km * self.options.temperature_lapse_rate, 1)

    def generate_precipitation(self):
        """
        Generate precipitation using wind patterns and orographic effects.

        Port of FMG's generatePrecipitation() from main.js:946-1106
        """
        logger.info("Generating precipitation")

        n_cells = len(self.graph.points)
        self.precipitation = np.zeros(n_cells, dtype=np.uint8)

        # Handle packed graphs that may not have cells_x/cells_y
        if hasattr(self.graph, "cells_x") and hasattr(self.graph, "cells_y"):
            cells_x = self.graph.cells_x
            cells_y = self.graph.cells_y
        else:
            # Estimate grid dimensions for packed graphs
            cells_x = int(np.sqrt(n_cells * 1.2))  # Slightly wider than square
            cells_y = int(n_cells / cells_x) if cells_x > 0 else int(np.sqrt(n_cells))

        # Modifiers
        cells_number_modifier = (n_cells / 10000) ** 0.25
        modifier = cells_number_modifier * self.options.precipitation_modifier

        # Wind sources by direction
        westerly = []
        easterly = []
        southerly = 0
        northerly = 0

        # Latitude modifiers for precipitation from options
        latitude_modifier = self.options.latitude_precipitation_modifiers

        # Define wind directions based on latitude
        for c in range(0, n_cells, cells_x):
            row_idx = c // cells_x
            y = self.graph.points[c][1]
            lat = self.map_coords.lat_n - (row_idx / cells_y) * self.map_coords.lat_t

            # Latitude band for precipitation modifier
            lat_band = int((abs(lat) - 1) / 5)
            lat_band = min(lat_band, len(latitude_modifier) - 1)
            lat_mod = latitude_modifier[lat_band]

            # Wind tier (0-5 from N to S) using configurable span
            wind_tier = int(abs(lat - 89) / self.options.wind_tier_span)
            wind_tier = min(wind_tier, len(self.options.winds) - 1)

            # Get wind directions for this tier
            is_west, is_east, is_north, is_south = self._get_wind_directions(wind_tier)

            if is_west:
                westerly.append((c, lat_mod, wind_tier))
            if is_east:
                east_index = min(c + cells_x - 1, n_cells - 1)
                easterly.append((east_index, lat_mod, wind_tier))

            if is_north:
                northerly += 1
            if is_south:
                southerly += 1

        # Pass winds across the map using configurable base precipitation
        if westerly:
            self._pass_wind(
                westerly, self.options.base_precipitation_west * modifier, 1, cells_x
            )
        if easterly:
            self._pass_wind(
                easterly, self.options.base_precipitation_west * modifier, -1, cells_x
            )

        # Vertical winds
        vert_total = southerly + northerly
        if northerly and vert_total > 0:
            band_n = int((abs(self.map_coords.lat_n) - 1) / 5)
            band_n = min(band_n, len(latitude_modifier) - 1)
            lat_mod_n = (
                latitude_modifier[band_n]
                if self.map_coords.lat_t <= 60
                else np.mean(latitude_modifier)
            )
            max_prec_n = (
                (northerly / vert_total)
                * self.options.base_precipitation_vertical
                * modifier
                * lat_mod_n
            )
            north_range = list(range(0, min(cells_x, n_cells)))  # Bound check
            self._pass_wind(north_range, max_prec_n, cells_x, cells_y)

        if southerly and vert_total > 0:
            band_s = int((abs(self.map_coords.lat_s) - 1) / 5)
            band_s = min(band_s, len(latitude_modifier) - 1)
            lat_mod_s = (
                latitude_modifier[band_s]
                if self.map_coords.lat_t <= 60
                else np.mean(latitude_modifier)
            )
            max_prec_s = (
                (southerly / vert_total)
                * self.options.base_precipitation_vertical
                * modifier
                * lat_mod_s
            )
            south_start = max(0, n_cells - cells_x)  # Bound check
            south_range = list(range(south_start, n_cells))
            self._pass_wind(south_range, max_prec_s, -cells_x, cells_y)

        # Store on graph
        self.graph.precipitation = self.precipitation

    def _get_wind_directions(self, tier: int) -> Tuple[bool, bool, bool, bool]:
        """Get wind direction flags for given tier."""
        angle = self.options.winds[tier]

        is_west = 40 < angle < 140
        is_east = 220 < angle < 320
        is_north = 100 < angle < 260
        is_south = angle > 280 or angle < 80

        return is_west, is_east, is_north, is_south

    def _pass_wind(self, source: List, max_prec: float, next_step: int, steps: int):
        """
        Simulate wind passing across terrain, depositing precipitation.

        Args:
            source: Starting cells for wind
            max_prec: Maximum precipitation amount
            next_step: Cell index increment per step
            steps: Number of steps to simulate
        """
        max_prec_init = max_prec
        max_passable_elevation = self.options.max_passable_elevation

        for first in source:
            # Handle tuples from westerly/easterly
            if isinstance(first, tuple):
                max_prec = min(max_prec_init * first[1], 255)
                first = first[0]

            # Initial humidity
            humidity = max_prec - self.graph.heights[first]
            if humidity <= 0:
                continue

            # Pass wind across terrain
            current = first
            for s in range(steps):
                if current < 0 or current >= len(self.graph.points):
                    break

                # Skip permafrost using configurable threshold
                if (
                    hasattr(self.graph, "temperatures")
                    and self.temperatures[current] < self.options.permafrost_threshold
                ):
                    current += next_step
                    continue

                # Water cell
                if self.graph.heights[current] < 20:
                    next_cell = current + next_step
                    if 0 <= next_cell < len(self.graph.points):
                        if self.graph.heights[next_cell] >= 20:
                            # Coastal precipitation using configurable range
                            precip = max(
                                humidity
                                / np.random.randint(*self.options.coastal_precip_range),
                                1,
                            )
                            self.precipitation[next_cell] += int(
                                min(precip, 255 - self.precipitation[next_cell])
                            )
                        else:
                            # Wind gains humidity over water using configurable values
                            humidity = min(
                                humidity
                                + self.options.water_humidity_gain
                                * self.options.precipitation_modifier,
                                max_prec,
                            )
                            water_precip = int(
                                self.options.water_precipitation
                                * self.options.precipitation_modifier
                            )
                            self.precipitation[current] += min(
                                water_precip, 255 - self.precipitation[current]
                            )
                    current += next_step
                    continue

                # Land cell
                next_cell = current + next_step
                if 0 <= next_cell < len(self.graph.points):
                    is_passable = (
                        self.graph.heights[next_cell] <= max_passable_elevation
                    )

                    if is_passable:
                        precipitation = self._get_precipitation(
                            humidity, current, next_step
                        )
                    else:
                        precipitation = (
                            humidity  # All humidity drops at impassable terrain
                        )

                    self.precipitation[current] += int(
                        min(precipitation, 255 - self.precipitation[current])
                    )

                    # Update humidity using configurable evaporation threshold
                    evaporation = (
                        1 if precipitation > self.options.evaporation_threshold else 0
                    )
                    humidity = (
                        max(0, humidity - precipitation + evaporation)
                        if is_passable
                        else 0
                    )
                else:
                    # Boundary - dump remaining humidity
                    self.precipitation[current] += int(
                        min(humidity, 255 - self.precipitation[current])
                    )
                    humidity = 0

                current += next_step

    def _get_precipitation(
        self, humidity: float, cell_idx: int, next_step: int
    ) -> float:
        """
        Calculate precipitation based on humidity and terrain.

        Includes orographic effects (increased precipitation on windward slopes).
        """
        # Normal precipitation loss using configurable divisor
        normal_loss = max(
            humidity
            / (
                self.options.precipitation_base_divisor
                * self.options.precipitation_modifier
            ),
            1,
        )

        # Orographic effect
        next_idx = cell_idx + next_step
        if 0 <= next_idx < len(self.graph.points):
            # Use int to avoid uint8 overflow
            height_diff = max(
                int(self.graph.heights[next_idx]) - int(self.graph.heights[cell_idx]), 0
            )
            terrain_mod = (
                self.graph.heights[next_idx] / self.options.terrain_mod_threshold
            ) ** 2  # Configurable mountain threshold
            orographic_precip = height_diff * terrain_mod
        else:
            orographic_precip = 0

        return min(normal_loss + orographic_precip, humidity)
