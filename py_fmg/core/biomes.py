"""
Biome classification system based on temperature and precipitation.

This module implements:
- Temperature/precipitation matrix classification (matching FMG)
- Geographic influence on biomes (coastal, riverine effects)
- Biome region generation and spatial optimization
"""

import structlog
import numpy as np
from typing import Dict, Optional, Set
from dataclasses import dataclass
from enum import IntEnum
from .voronoi_graph import (
    VoronoiGraph,
    build_cell_connectivity,
    build_cell_vertices,
    build_vertex_connectivity,
)

logger = structlog.get_logger()


class BiomeType(IntEnum):
    """Biome types matching FMG classification."""

    OCEAN = 0
    LAKE = 1
    RIVER = 2
    WETLAND = 3
    GLACIER = 4
    TUNDRA = 5
    TAIGA = 6
    TEMPERATE_DECIDUOUS_FOREST = 7
    TEMPERATE_RAINFOREST = 8
    TEMPERATE_GRASSLAND = 9
    MEDITERRANEAN = 10
    DESERT = 11
    HOT_DESERT = 12
    SAVANNA = 13
    TROPICAL_SEASONAL_FOREST = 14
    TROPICAL_RAINFOREST = 15
    MANGROVE = 16
    ALPINE = 17


# Biome names for display
BIOME_NAMES = {
    BiomeType.OCEAN: "Ocean",
    BiomeType.LAKE: "Lake",
    BiomeType.RIVER: "River",
    BiomeType.WETLAND: "Wetland",
    BiomeType.GLACIER: "Glacier",
    BiomeType.TUNDRA: "Tundra",
    BiomeType.TAIGA: "Taiga",
    BiomeType.TEMPERATE_DECIDUOUS_FOREST: "Temperate Deciduous Forest",
    BiomeType.TEMPERATE_RAINFOREST: "Temperate Rainforest",
    BiomeType.TEMPERATE_GRASSLAND: "Temperate Grassland",
    BiomeType.MEDITERRANEAN: "Mediterranean",
    BiomeType.DESERT: "Desert",
    BiomeType.HOT_DESERT: "Hot Desert",
    BiomeType.SAVANNA: "Savanna",
    BiomeType.TROPICAL_SEASONAL_FOREST: "Tropical Seasonal Forest",
    BiomeType.TROPICAL_RAINFOREST: "Tropical Rainforest",
    BiomeType.MANGROVE: "Mangrove",
    BiomeType.ALPINE: "Alpine",
}


@dataclass
class BiomeOptions:
    """Biome classification options."""

    coastal_effect_distance: float = 3.0  # Distance for coastal biome effects
    river_effect_distance: float = 2.0  # Distance for river biome effects
    wetland_threshold: float = 0.8  # Threshold for wetland formation
    alpine_height_threshold: int = 70  # Height threshold for alpine biomes
    glacier_temperature_threshold: int = -10  # Temperature threshold for glaciers


@dataclass
class BiomeRegion:
    """Represents a contiguous biome region."""

    id: int
    biome_type: BiomeType
    cells: Set[int]
    area: float
    center_cell: int


class BiomeClassifier:
    """Handles biome classification based on climate and geography."""

    def __init__(self, graph:VoronoiGraph, options: Optional[BiomeOptions] = None):
        """
        Initialize biome classifier.

        Args:
            graph: VoronoiGraph with climate and hydrology data
            options: Biome classification options
        """
        self.graph = graph
        self.options = options or BiomeOptions()

        # Ensure required tile events are initialized
        self._ensure_tile_events()

        # Classification results
        self.biomes = None  # Biome type for each cell
        self.biome_regions = []  # List of BiomeRegion objects

        # Temperature/precipitation classification matrix
        # Based on FMG's biome matrix (temperature bands vs precipitation bands)
        self._init_biome_matrix()

    # TODO some sort of decision algorithm based on given infos -> collapse way
    def get_biome_properties(self, biome_id):
        return biome_id
    
    def get_biome_id(self, biome_type:BiomeType, temperature:int, height:float, ver:bool):
        return 

    def _ensure_tile_events(self):
        """Ensure required tile events are initialized in the graph."""
        required_events = ['biome_regions', 'biomes']
        
        for event in required_events:
            if not self.graph.has_tile_data(event):
                # Initialize with appropriate default values
                if event == 'biome_regions':
                    self.graph.set_tile_data(event, [])
                    logger.info(f"Initialized tile event '{event}' with default list")
                elif event == 'biomes':
                    default_value = np.zeros(len(self.graph.cell_neighbors), dtype=int)
                    self.graph.set_tile_data(event, default_value)
                    logger.info(f"Initialized tile event '{event}' with default array")

    def _validate_prerequisites(self):
        """Validate that required data exists before biome classification."""
        if not self.graph.heights is not None or self.graph.heights is None:
            raise ValueError("Heights must be calculated before biome classification")
        
        # Check for climate data (temperatures and precipitation)
        if not self.graph.has_tile_data('temperatures'):
            logger.warning("No temperature data found, using default values")
        
        if not self.graph.has_tile_data('precipitation'):
            logger.warning("No precipitation data found, using default values")
        
        logger.info("Biome classification prerequisites validated")

    def _init_biome_matrix(self):
        """
        Initialize the temperature/precipitation biome classification matrix.

        This replicates FMG's biome classification logic.
        Temperature bands: 0=very cold, 1=cold, 2=cool, 3=temperate, 4=warm, 5=hot
        Precipitation bands: 0=very dry, 1=dry, 2=moderate, 3=wet, 4=very wet
        """
        # Biome matrix [temperature_band][precipitation_band]
        self.biome_matrix = [
            # Very Cold (-30 to -10°C)
            [
                BiomeType.GLACIER,
                BiomeType.GLACIER,
                BiomeType.TUNDRA,
                BiomeType.TUNDRA,
                BiomeType.TUNDRA,
            ],
            # Cold (-10 to 5°C)
            [
                BiomeType.TUNDRA,
                BiomeType.TUNDRA,
                BiomeType.TAIGA,
                BiomeType.TAIGA,
                BiomeType.TAIGA,
            ],
            # Cool (5 to 15°C)
            [
                BiomeType.DESERT,
                BiomeType.TEMPERATE_GRASSLAND,
                BiomeType.TEMPERATE_DECIDUOUS_FOREST,
                BiomeType.TEMPERATE_DECIDUOUS_FOREST,
                BiomeType.TEMPERATE_RAINFOREST,
            ],
            # Temperate (15 to 25°C)
            [
                BiomeType.DESERT,
                BiomeType.TEMPERATE_GRASSLAND,
                BiomeType.TEMPERATE_DECIDUOUS_FOREST,
                BiomeType.TEMPERATE_DECIDUOUS_FOREST,
                BiomeType.TEMPERATE_RAINFOREST,
            ],
            # Warm (25 to 35°C)
            [
                BiomeType.HOT_DESERT,
                BiomeType.SAVANNA,
                BiomeType.TROPICAL_SEASONAL_FOREST,
                BiomeType.TROPICAL_RAINFOREST,
                BiomeType.TROPICAL_RAINFOREST,
            ],
            # Hot (35°C+)
            [
                BiomeType.HOT_DESERT,
                BiomeType.HOT_DESERT,
                BiomeType.SAVANNA,
                BiomeType.TROPICAL_SEASONAL_FOREST,
                BiomeType.TROPICAL_RAINFOREST,
            ],
        ]

        # Temperature band thresholds (°C)
        self.temp_thresholds = [-10, 5, 15, 25, 35]

        # Precipitation band thresholds (mm/year equivalent, scaled to 0-255)
        self.precip_thresholds = [20, 50, 100, 180]

    def classify_biomes(self):
        """
        Classify biomes for all cells based on climate data.
        """
        logger.info("Classifying biomes")

        n_cells = len(self.graph.points)
        self.biomes = np.full(n_cells, BiomeType.OCEAN, dtype=np.uint8)

        for i in range(n_cells):
            self.biomes[i] = self._classify_cell_biome(i)

        # Apply geographic influences
        self._apply_geographic_influences()

        # Store on graph
        self.graph.biomes = self.biomes

        logger.info(
            "Biome classification completed", unique_biomes=len(np.unique(self.biomes))
        )

    def _classify_cell_biome(self, cell_idx: int) -> BiomeType:
        """
        Classify biome for a single cell.

        Args:
            cell_idx: Cell index

        Returns:
            BiomeType for the cell
        """
        # Water cells
        if self.graph.heights[cell_idx] < 20:
            return BiomeType.OCEAN

        # Get climate data using tile_events system
        temperature = (
            self.graph.temperatures[cell_idx]
            if self.graph.temperatures is not None and cell_idx < len(self.graph.temperatures)
            else 15
        )
        precipitation = (
            self.graph.precipitation[cell_idx]
            if self.graph.precipitation is not None and cell_idx < len(self.graph.precipitation)
            else 100
        )
        height = self.graph.heights[cell_idx]

        # Special cases first

        # Alpine biome for high elevations
        if height >= self.options.alpine_height_threshold:
            return BiomeType.ALPINE

        # Glacier for very cold areas
        if temperature <= self.options.glacier_temperature_threshold:
            return BiomeType.GLACIER

        # Determine temperature and precipitation bands
        temp_band = self._get_temperature_band(temperature)
        precip_band = self._get_precipitation_band(precipitation)

        # Look up biome in matrix
        return self.biome_matrix[temp_band][precip_band]

    def _get_temperature_band(self, temperature: float) -> int:
        """Get temperature band index (0-5)."""
        for i, threshold in enumerate(self.temp_thresholds):
            if temperature < threshold:
                return i
        return len(self.temp_thresholds)  # Hottest band

    def _get_precipitation_band(self, precipitation: float) -> int:
        """Get precipitation band index (0-4)."""
        for i, threshold in enumerate(self.precip_thresholds):
            if precipitation < threshold:
                return i
        return len(self.precip_thresholds)  # Wettest band

    def _apply_geographic_influences(self):
        """
        Apply geographic influences on biome classification.

        This includes coastal effects, river influences, and wetland formation.
        """
        logger.info("Applying geographic influences")

        # Apply coastal influences
        self._apply_coastal_influences()

        # Apply river influences
        self._apply_river_influences()

        # Form wetlands
        self._form_wetlands()

        # Form mangroves
        self._form_mangroves()

    def _apply_coastal_influences(self):
        """Apply coastal influences on biomes."""
        if (
            not self.graph.distance_field is not None
            or self.graph.distance_field is None
        ):
            return

        for i in range(len(self.graph.points)):
            if self.graph.heights[i] >= 20:  # Land cell
                distance_to_coast = abs(self.graph.distance_field[i])

                if distance_to_coast <= self.options.coastal_effect_distance:
                    # Coastal areas are more humid
                    current_biome = self.biomes[i]

                    # Mediterranean climate near coasts in temperate zones
                    if (
                        self.graph.temperatures[i] > 15
                        and self.graph.temperatures[i] < 25
                        and current_biome
                        in [BiomeType.TEMPERATE_GRASSLAND, BiomeType.DESERT]
                    ):
                        self.biomes[i] = BiomeType.MEDITERRANEAN

    def _apply_river_influences(self):
        """Apply river influences on biomes."""
        if not self.graph.rivers is not None:
            return

        river_cells = set()
        for river in self.graph.rivers:
            river_cells.update(river.cells)

        # Mark river cells
        for cell_idx in river_cells:
            if self.graph.heights[cell_idx] >= 20:  # Land river
                self.biomes[cell_idx] = BiomeType.RIVER

        # Influence nearby cells
        for cell_idx in river_cells:
            for neighbor_idx in self.graph.cell_neighbors[cell_idx]:
                if (
                    self.graph.heights[neighbor_idx] >= 20
                    and neighbor_idx not in river_cells
                ):

                    # Rivers increase local humidity
                    current_biome = self.biomes[neighbor_idx]

                    # Convert dry biomes to more humid variants
                    if current_biome == BiomeType.DESERT:
                        self.biomes[neighbor_idx] = BiomeType.TEMPERATE_GRASSLAND
                    elif current_biome == BiomeType.HOT_DESERT:
                        self.biomes[neighbor_idx] = BiomeType.SAVANNA

    def _form_wetlands(self):
        """Form wetlands in appropriate locations."""
        if not self.graph.water_flux is not None:
            return

        # Find areas with high water accumulation but not rivers
        for i in range(len(self.graph.points)):
            if (
                self.graph.heights[i] >= 20  # Land
                and self.biomes[i] != BiomeType.RIVER
                and self.graph.water_flux is not None
            ):

                # High water flux indicates potential wetland
                water_flux = self.graph.water_flux[i]
                max_flux = np.max(self.graph.water_flux)

                if water_flux > max_flux * self.options.wetland_threshold:
                    # Check if it's not already a major river
                    is_major_river = False
                    if self.graph.rivers is not None:
                        for river in self.graph.rivers:
                            if i in river.cells and river.flow > 100:
                                is_major_river = True
                                break

                    if not is_major_river:
                        self.biomes[i] = BiomeType.WETLAND

    def _form_mangroves(self):
        """Form mangroves in tropical coastal areas."""
        if (
            not self.graph.distance_field is not None
            or self.graph.distance_field is None
        ):
            return

        for i in range(len(self.graph.points)):
            if (
                self.graph.heights[i] >= 20  # Land
                and abs(self.graph.distance_field[i]) <= 1.0  # Very close to coast
                and self.graph.temperatures[i] > 20
            ):  # Tropical temperature

                # Form mangroves in tropical coastal areas
                self.biomes[i] = BiomeType.MANGROVE

    def generate_biome_regions(self):
        """
        Generate contiguous biome regions for spatial optimization.

        This groups adjacent cells of the same biome into regions.
        """
        logger.info("Generating biome regions")

        if self.biomes is None:
            self.classify_biomes()

        self.biome_regions = []
        processed = set()
        region_id = 0

        for i in range(len(self.graph.points)):
            if i in processed:
                continue

            # Start a new region
            biome_type = BiomeType(self.biomes[i])
            region_cells = self._find_connected_biome_cells(i, biome_type, processed)

            if len(region_cells) > 0:
                # Calculate region properties
                center_cell = self._find_region_center(region_cells)
                area = len(region_cells) * (self.graph.spacing**2)

                region = BiomeRegion(
                    id=region_id,
                    biome_type=biome_type,
                    cells=region_cells,
                    area=area,
                    center_cell=center_cell,
                )

                self.biome_regions.append(region)
                processed.update(region_cells)
                region_id += 1

        # Store on graph
        self.graph.biome_regions = self.biome_regions

        logger.info("Biome regions generated", count=len(self.biome_regions))

    def _find_connected_biome_cells(
        self, start_cell: int, biome_type: BiomeType, processed: Set[int]
    ) -> Set[int]:
        """
        Find all connected cells of the same biome type.

        Args:
            start_cell: Starting cell index
            biome_type: Biome type to match
            processed: Set of already processed cells

        Returns:
            Set of connected cell indices
        """
        region_cells = set()
        queue = [start_cell]

        while queue:
            cell = queue.pop(0)

            if cell in region_cells or cell in processed:
                continue

            if BiomeType(self.biomes[cell]) == biome_type:
                region_cells.add(cell)

                # Add neighbors to queue
                for neighbor in self.graph.cell_neighbors[cell]:
                    if (
                        neighbor not in region_cells
                        and neighbor not in processed
                        and BiomeType(self.biomes[neighbor]) == biome_type
                    ):
                        queue.append(neighbor)

        return region_cells

    def _find_region_center(self, region_cells: Set[int]) -> int:
        """
        Find the center cell of a region.

        Args:
            region_cells: Set of cell indices in the region

        Returns:
            Cell index closest to the region centroid
        """
        if not region_cells:
            return -1

        # Calculate centroid
        points = [self.graph.points[cell] for cell in region_cells]
        centroid_x = np.mean([p[0] for p in points])
        centroid_y = np.mean([p[1] for p in points])

        # Find closest cell to centroid
        min_distance = float("inf")
        center_cell = next(iter(region_cells))

        for cell in region_cells:
            x, y = self.graph.points[cell]
            distance = (x - centroid_x) ** 2 + (y - centroid_y) ** 2

            if distance < min_distance:
                min_distance = distance
                center_cell = cell

        return center_cell

    def run_full_classification(self):
        """
        Run the complete biome classification pipeline.

        This executes all steps in the correct order:
        1. Classify biomes based on climate
        2. Apply geographic influences
        3. Generate biome regions
        """
        logger.info("Starting full biome classification")

        self.classify_biomes()
        self.generate_biome_regions()

        logger.info(
            "Biome classification completed",
            biomes=len(np.unique(self.biomes)),
            regions=len(self.biome_regions),
        )

    def get_biome_statistics(self) -> Dict[str, int]:
        """
        Get statistics about biome distribution.

        Returns:
            Dictionary with biome names and cell counts
        """
        if self.biomes is None:
            return {}

        stats = {}
        unique_biomes, counts = np.unique(self.biomes, return_counts=True)

        for biome_id, count in zip(unique_biomes, counts):
            biome_type = BiomeType(biome_id)
            biome_name = BIOME_NAMES[biome_type]
            stats[biome_name] = count

        return stats

    def get_biome_habitability(self, biome_type: BiomeType) -> int:
        """
        Get habitability score for a biome type.

        Returns:
            Habitability score (0-100, where 100 is most habitable)
        """
        # Habitability data from FMG
        habitability_map = {
            BiomeType.OCEAN: 0,
            BiomeType.LAKE: 20,  # Some lake shores are habitable
            BiomeType.RIVER: 30,  # River valleys are moderately habitable
            BiomeType.WETLAND: 12,
            BiomeType.GLACIER: 0,
            BiomeType.TUNDRA: 4,
            BiomeType.TAIGA: 12,
            BiomeType.TEMPERATE_DECIDUOUS_FOREST: 100,  # Most habitable
            BiomeType.TEMPERATE_RAINFOREST: 90,
            BiomeType.TEMPERATE_GRASSLAND: 30,
            BiomeType.MEDITERRANEAN: 80,  # Good habitability
            BiomeType.DESERT: 4,
            BiomeType.HOT_DESERT: 4,
            BiomeType.SAVANNA: 22,
            BiomeType.TROPICAL_SEASONAL_FOREST: 50,
            BiomeType.TROPICAL_RAINFOREST: 80,
            BiomeType.MANGROVE: 10,
            BiomeType.ALPINE: 10,
        }

        return habitability_map.get(biome_type, 50)  # Default to 50 if not found

    def get_biome_movement_cost(self, biome_type: BiomeType) -> int:
        """
        Get movement cost for a biome type.

        Returns:
            Movement cost (higher = harder to traverse)
        """
        # Movement cost data from FMG
        movement_cost_map = {
            BiomeType.OCEAN: 10,  # Naval movement
            BiomeType.LAKE: 10,  # Naval movement
            BiomeType.RIVER: 20,  # Can follow rivers
            BiomeType.WETLAND: 150,
            BiomeType.GLACIER: 5000,  # Nearly impassable
            BiomeType.TUNDRA: 1000,
            BiomeType.TAIGA: 200,
            BiomeType.TEMPERATE_DECIDUOUS_FOREST: 70,
            BiomeType.TEMPERATE_RAINFOREST: 90,
            BiomeType.TEMPERATE_GRASSLAND: 50,
            BiomeType.MEDITERRANEAN: 60,
            BiomeType.DESERT: 150,
            BiomeType.HOT_DESERT: 200,
            BiomeType.SAVANNA: 60,
            BiomeType.TROPICAL_SEASONAL_FOREST: 70,
            BiomeType.TROPICAL_RAINFOREST: 80,
            BiomeType.MANGROVE: 100,
            BiomeType.ALPINE: 300,
        }

        return movement_cost_map.get(biome_type, 100)  # Default to 100 if not found

    def get_biome_colors(self) -> Dict[BiomeType, str]:
        """
        Get color mapping for all biome types.

        Returns:
            Dictionary mapping BiomeType to hex color string
        """
        return {
            BiomeType.OCEAN: "#466eab",
            BiomeType.LAKE: "#6ba9cb",
            BiomeType.RIVER: "#4682b4",
            BiomeType.WETLAND: "#0b9131",
            BiomeType.GLACIER: "#d5e7eb",
            BiomeType.TUNDRA: "#96784b",
            BiomeType.TAIGA: "#4b6b32",
            BiomeType.TEMPERATE_DECIDUOUS_FOREST: "#29bc56",
            BiomeType.TEMPERATE_RAINFOREST: "#409c43",
            BiomeType.TEMPERATE_GRASSLAND: "#c8d68f",
            BiomeType.MEDITERRANEAN: "#daa520",
            BiomeType.DESERT: "#b5b887",
            BiomeType.HOT_DESERT: "#fbe79f",
            BiomeType.SAVANNA: "#d2d082",
            BiomeType.TROPICAL_SEASONAL_FOREST: "#b6d95d",
            BiomeType.TROPICAL_RAINFOREST: "#7dcb35",
            BiomeType.MANGROVE: "#2f4f2f",
            BiomeType.ALPINE: "#a0a0a0",
        }
