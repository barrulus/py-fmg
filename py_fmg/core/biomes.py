"""
Biome Classification System

Ports the biome assignment algorithm from Fantasy Map Generator (FMG).
Classifies terrain based on temperature, moisture, and special conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BiomeData:
    """Container for biome properties and metadata"""

    names: List[str]
    colors: List[str]
    habitability: List[int]
    icons_density: List[int]
    icons: List[Dict[str, int]]
    movement_cost: List[int]
    biome_matrix: np.ndarray


class BiomeClassifier:
    """
    Biome classification system based on temperature-moisture matrix lookup
    with special condition overrides.
    """

    MIN_LAND_HEIGHT = 20

    def __init__(self) -> None:
        self.biome_data = self._create_default_biome_data()

    def _create_default_biome_data(self) -> BiomeData:
        """Create default biome data matching FMG specification"""
        names = [
            "Marine",
            "Hot desert",
            "Cold desert",
            "Savanna",
            "Grassland",
            "Tropical seasonal forest",
            "Temperate deciduous forest",
            "Tropical rainforest",
            "Temperate rainforest",
            "Taiga",
            "Tundra",
            "Glacier",
            "Wetland",
        ]

        colors = [
            "#466eab",  # Marine
            "#fbe79f",  # Hot desert
            "#b5b887",  # Cold desert
            "#d2d082",  # Savanna
            "#c8d68f",  # Grassland
            "#b6d95d",  # Tropical seasonal forest
            "#29bc56",  # Temperate deciduous forest
            "#7dcb35",  # Tropical rainforest
            "#409c43",  # Temperate rainforest
            "#4b6b32",  # Taiga
            "#96784b",  # Tundra
            "#d5e7eb",  # Glacier
            "#0b9131",  # Wetland
        ]

        habitability = [0, 4, 10, 22, 30, 50, 100, 80, 90, 12, 4, 0, 12]
        icons_density = [0, 3, 2, 120, 120, 120, 120, 150, 150, 100, 5, 0, 250]

        # Icon weights as dictionaries (will be expanded to arrays when needed)
        icons = [
            {},  # Marine
            {"dune": 3, "cactus": 6, "deadTree": 1},  # Hot desert
            {"dune": 9, "deadTree": 1},  # Cold desert
            {"acacia": 1, "grass": 9},  # Savanna
            {"grass": 1},  # Grassland
            {"acacia": 8, "palm": 1},  # Tropical seasonal forest
            {"deciduous": 1},  # Temperate deciduous forest
            {"acacia": 5, "palm": 3, "deciduous": 1, "swamp": 1},  # Tropical rainforest
            {"deciduous": 6, "swamp": 1},  # Temperate rainforest
            {"conifer": 1},  # Taiga
            {"grass": 1},  # Tundra
            {},  # Glacier
            {"swamp": 1},  # Wetland
        ]

        movement_cost = [10, 200, 150, 60, 50, 70, 70, 80, 90, 200, 1000, 5000, 150]

        # Biome matrix: 5x26 temperature-moisture lookup table
        # Rows: moisture bands (0-4), Columns: temperature bands (0-25)
        # hot ↔ cold [>19°C; <-4°C]; dry ↕ wet
        biome_matrix = np.array(
            [
                # Moisture band 0 (driest)
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    10,
                ],
                # Moisture band 1
                [
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    9,
                    9,
                    9,
                    9,
                    10,
                    10,
                    10,
                ],
                # Moisture band 2
                [
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    9,
                    9,
                    9,
                    9,
                    9,
                    10,
                    10,
                    10,
                ],
                # Moisture band 3
                [
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    10,
                    10,
                    10,
                ],
                # Moisture band 4 (wettest)
                [
                    7,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    10,
                    10,
                ],
            ],
            dtype=np.uint8,
        )

        return BiomeData(
            names=names,
            colors=colors,
            habitability=habitability,
            icons_density=icons_density,
            icons=icons,
            movement_cost=movement_cost,
            biome_matrix=biome_matrix,
        )

    def calculate_moisture(
        self,
        cell_id: int,
        precipitation: np.ndarray,
        heights: np.ndarray,
        river_flux: Optional[np.ndarray] = None,
        has_river: Optional[np.ndarray] = None,
        neighbors: Optional[List[List[int]]] = None,
        grid_reference: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate moisture for a single cell based on precipitation, rivers, neighbors.

        Args:
            cell_id: Cell index
            precipitation: Grid precipitation values
            heights: Cell heights
            river_flux: River flux values (optional)
            has_river: Boolean array indicating river presence (optional)
            neighbors: Neighbor lists for each cell (optional)
            grid_reference: Mapping from cell to grid index (optional)

        Returns:
            Calculated moisture value
        """
        if heights[cell_id] < self.MIN_LAND_HEIGHT:
            return 0.0

        # Base moisture from precipitation
        if grid_reference is not None:
            moisture = precipitation[grid_reference[cell_id]]
        else:
            moisture = precipitation[cell_id]

        # Add river bonus if present
        if has_river is not None and has_river[cell_id]:
            if river_flux is not None:
                river_bonus = max(river_flux[cell_id] / 10.0, 2.0)
            else:
                river_bonus = 2.0
            moisture += river_bonus

        # Average with neighbor moisture for smooth transitions
        if neighbors is not None and cell_id < len(neighbors):
            neighbor_moisture = []
            for neighbor_id in neighbors[cell_id]:
                if (
                    neighbor_id < len(heights)
                    and heights[neighbor_id] >= self.MIN_LAND_HEIGHT
                ):
                    if grid_reference is not None:
                        neighbor_moisture.append(
                            precipitation[grid_reference[neighbor_id]]
                        )
                    else:
                        neighbor_moisture.append(precipitation[neighbor_id])

            if neighbor_moisture:
                all_moisture = neighbor_moisture + [moisture]
                moisture = np.mean(all_moisture)

        return float(round(4.0 + moisture, 1))

    def calculate_moisture_vectorized(
        self,
        precipitation: np.ndarray,
        heights: np.ndarray,
        river_flux: Optional[np.ndarray] = None,
        has_river: Optional[np.ndarray] = None,
        neighbors: Optional[List[List[int]]] = None,
        grid_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vectorized moisture calculation for all cells.

        Args:
            precipitation: Grid precipitation values
            heights: Cell heights
            river_flux: River flux values (optional)
            has_river: Boolean array indicating river presence (optional)
            neighbors: Neighbor lists for each cell (optional)
            grid_reference: Mapping from cell to grid index (optional)

        Returns:
            Array of moisture values for all cells
        """
        n_cells = len(heights)
        moisture = np.zeros(n_cells, dtype=np.float32)

        # Water cells have 0 moisture
        land_mask = heights >= self.MIN_LAND_HEIGHT

        # Base moisture from precipitation
        if grid_reference is not None:
            moisture[land_mask] = precipitation[grid_reference[land_mask]]
        else:
            moisture[land_mask] = precipitation[land_mask]

        # Add river bonus
        if has_river is not None:
            river_mask = land_mask & has_river
            if river_flux is not None:
                river_bonus = np.maximum(river_flux[river_mask] / 10.0, 2.0)
            else:
                river_bonus = 2.0
            moisture[river_mask] += river_bonus

        # Neighbor averaging (simplified - could be optimized further)
        if neighbors is not None:
            for cell_id in range(n_cells):
                if land_mask[cell_id]:
                    neighbor_moisture = []
                    for neighbor_id in neighbors[cell_id]:
                        if (
                            neighbor_id < n_cells
                            and heights[neighbor_id] >= self.MIN_LAND_HEIGHT
                        ):
                            if grid_reference is not None:
                                neighbor_moisture.append(
                                    precipitation[grid_reference[neighbor_id]]
                                )
                            else:
                                neighbor_moisture.append(precipitation[neighbor_id])

                    if neighbor_moisture:
                        all_moisture = neighbor_moisture + [moisture[cell_id]]
                        moisture[cell_id] = np.mean(all_moisture)

        # Add baseline offset and round
        moisture[land_mask] = np.round(4.0 + moisture[land_mask], 1)

        return moisture

    def is_wetland(self, moisture: float, temperature: float, height: float) -> bool:
        """
        Check if conditions meet wetland biome criteria.

        Args:
            moisture: Cell moisture value
            temperature: Cell temperature
            height: Cell height

        Returns:
            True if cell should be classified as wetland
        """
        if temperature <= -2:
            return False  # Too cold

        if moisture > 40 and height < 25:
            return True  # Near coast

        if moisture > 24 and height > 24 and height < 60:
            return True  # Off coast

        return False

    def get_biome_id(
        self, moisture: float, temperature: float, height: float, has_river: bool
    ) -> int:
        """
        Get biome ID for a single cell based on environmental conditions.

        Args:
            moisture: Cell moisture value
            temperature: Cell temperature
            height: Cell height
            has_river: Whether cell contains a river

        Returns:
            Biome ID (0-12)
        """
        # Special condition overrides (order matters!)
        if height < self.MIN_LAND_HEIGHT:
            return 0  # Marine

        if temperature < -5:
            return 11  # Permafrost/Glacier

        if temperature >= 25 and not has_river and moisture < 8:
            return 1  # Hot desert

        if self.is_wetland(moisture, temperature, height):
            return 12  # Wetland

        # Use biome matrix for normal classification
        moisture_band = min(int(moisture / 5), 4)  # [0-4]
        temperature_band = min(max(int(20 - temperature), 0), 25)  # [0-25]

        return int(self.biome_data.biome_matrix[moisture_band, temperature_band])

    def classify_biomes(
        self,
        temperatures: np.ndarray,
        precipitation: np.ndarray,
        heights: np.ndarray,
        river_flux: Optional[np.ndarray] = None,
        has_river: Optional[np.ndarray] = None,
        neighbors: Optional[List[List[int]]] = None,
        grid_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Classify biomes for all cells using vectorized operations.

        Args:
            temperatures: Cell temperature values
            precipitation: Grid precipitation values
            heights: Cell heights
            river_flux: River flux values (optional)
            has_river: Boolean array indicating river presence (optional)
            neighbors: Neighbor lists for each cell (optional)
            grid_reference: Mapping from cell to grid index (optional)

        Returns:
            Array of biome IDs for all cells
        """
        n_cells = len(heights)
        biomes = np.zeros(n_cells, dtype=np.uint8)

        # Calculate moisture for all cells
        moisture = self.calculate_moisture_vectorized(
            precipitation, heights, river_flux, has_river, neighbors, grid_reference
        )

        # Apply special conditions with boolean indexing
        # Marine (water)
        marine_mask = heights < self.MIN_LAND_HEIGHT
        biomes[marine_mask] = 0

        # Permafrost (very cold)
        permafrost_mask = (temperatures < -5) & ~marine_mask
        biomes[permafrost_mask] = 11

        # Hot desert (hot, dry, no river)
        if has_river is not None:
            hot_desert_mask = (
                (temperatures >= 25)
                & ~has_river
                & (moisture < 8)
                & ~marine_mask
                & ~permafrost_mask
            )
        else:
            hot_desert_mask = (
                (temperatures >= 25) & (moisture < 8) & ~marine_mask & ~permafrost_mask
            )
        biomes[hot_desert_mask] = 1

        # Wetland conditions
        wetland_mask = np.zeros(n_cells, dtype=bool)
        land_mask = ~marine_mask & ~permafrost_mask & ~hot_desert_mask

        for i in range(n_cells):
            if land_mask[i] and self.is_wetland(
                moisture[i], temperatures[i], heights[i]
            ):
                wetland_mask[i] = True
        biomes[wetland_mask] = 12

        # Use matrix for remaining cells
        normal_mask = ~marine_mask & ~permafrost_mask & ~hot_desert_mask & ~wetland_mask

        if np.any(normal_mask):
            moisture_bands = np.clip((moisture[normal_mask] / 5).astype(int), 0, 4)
            temperature_bands = np.clip(
                (20 - temperatures[normal_mask]).astype(int), 0, 25
            )
            biomes[normal_mask] = self.biome_data.biome_matrix[
                moisture_bands, temperature_bands
            ]

        return biomes

    def get_biome_name(self, biome_id: int) -> str:
        """Get biome name by ID"""
        if 0 <= biome_id < len(self.biome_data.names):
            return str(self.biome_data.names[biome_id])
        return "Unknown"

    def get_biome_color(self, biome_id: int) -> str:
        """Get biome color by ID"""
        if 0 <= biome_id < len(self.biome_data.colors):
            return str(self.biome_data.colors[biome_id])
        return "#000000"

    def get_biome_properties(self, biome_id: int) -> Dict:
        """Get all properties for a biome ID"""
        if not (0 <= biome_id < len(self.biome_data.names)):
            return {}

        return {
            "id": biome_id,
            "name": self.biome_data.names[biome_id],
            "color": self.biome_data.colors[biome_id],
            "habitability": self.biome_data.habitability[biome_id],
            "icons_density": self.biome_data.icons_density[biome_id],
            "icons": self.biome_data.icons[biome_id],
            "movement_cost": self.biome_data.movement_cost[biome_id],
        }
