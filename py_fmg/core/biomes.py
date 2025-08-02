"""
Biome classification system based on temperature and precipitation.

This module implements:
- Temperature/precipitation matrix classification (matching FMG)
- Geographic influence on biomes (coastal, riverine effects)
- Biome region generation and spatial optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import IntEnum
import structlog

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
    BiomeType.ALPINE: "Alpine"
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
    
    def __init__(self, graph, options: Optional[BiomeOptions] = None):
        """
        Initialize biome classifier.
        
        Args:
            graph: VoronoiGraph with climate and hydrology data
            options: Biome classification options
        """
        self.graph = graph
        self.options = options or BiomeOptions()
        
        # Classification results
        self.biomes = None  # Biome type for each cell
        self.biome_regions = []  # List of BiomeRegion objects
        
        # Temperature/precipitation classification matrix
        # Based on FMG's biome matrix (temperature bands vs precipitation bands)
        self._init_biome_matrix()
        
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
            [BiomeType.GLACIER, BiomeType.GLACIER, BiomeType.TUNDRA, BiomeType.TUNDRA, BiomeType.TUNDRA],
            # Cold (-10 to 5°C)  
            [BiomeType.TUNDRA, BiomeType.TUNDRA, BiomeType.TAIGA, BiomeType.TAIGA, BiomeType.TAIGA],
            # Cool (5 to 15°C)
            [BiomeType.DESERT, BiomeType.TEMPERATE_GRASSLAND, BiomeType.TEMPERATE_DECIDUOUS_FOREST, 
             BiomeType.TEMPERATE_DECIDUOUS_FOREST, BiomeType.TEMPERATE_RAINFOREST],
            # Temperate (15 to 25°C)
            [BiomeType.DESERT, BiomeType.TEMPERATE_GRASSLAND, BiomeType.TEMPERATE_DECIDUOUS_FOREST,
             BiomeType.TEMPERATE_DECIDUOUS_FOREST, BiomeType.TEMPERATE_RAINFOREST],
            # Warm (25 to 35°C)
            [BiomeType.HOT_DESERT, BiomeType.SAVANNA, BiomeType.TROPICAL_SEASONAL_FOREST,
             BiomeType.TROPICAL_RAINFOREST, BiomeType.TROPICAL_RAINFOREST],
            # Hot (35°C+)
            [BiomeType.HOT_DESERT, BiomeType.HOT_DESERT, BiomeType.SAVANNA,
             BiomeType.TROPICAL_SEASONAL_FOREST, BiomeType.TROPICAL_RAINFOREST]
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
        
        logger.info("Biome classification completed",
                   unique_biomes=len(np.unique(self.biomes)))
        
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
            
        # Get climate data
        temperature = self.graph.temperatures[cell_idx] if hasattr(self.graph, 'temperatures') else 15
        precipitation = self.graph.precipitation[cell_idx] if hasattr(self.graph, 'precipitation') else 100
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
        if not hasattr(self.graph, 'distance_field') or self.graph.distance_field is None:
            return
            
        for i in range(len(self.graph.points)):
            if self.graph.heights[i] >= 20:  # Land cell
                distance_to_coast = abs(self.graph.distance_field[i])
                
                if distance_to_coast <= self.options.coastal_effect_distance:
                    # Coastal areas are more humid
                    current_biome = self.biomes[i]
                    
                    # Mediterranean climate near coasts in temperate zones
                    if (self.graph.temperatures[i] > 15 and 
                        self.graph.temperatures[i] < 25 and
                        current_biome in [BiomeType.TEMPERATE_GRASSLAND, BiomeType.DESERT]):
                        self.biomes[i] = BiomeType.MEDITERRANEAN
                        
    def _apply_river_influences(self):
        """Apply river influences on biomes."""
        if not hasattr(self.graph, 'rivers'):
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
                if (self.graph.heights[neighbor_idx] >= 20 and 
                    neighbor_idx not in river_cells):
                    
                    # Rivers increase local humidity
                    current_biome = self.biomes[neighbor_idx]
                    
                    # Convert dry biomes to more humid variants
                    if current_biome == BiomeType.DESERT:
                        self.biomes[neighbor_idx] = BiomeType.TEMPERATE_GRASSLAND
                    elif current_biome == BiomeType.HOT_DESERT:
                        self.biomes[neighbor_idx] = BiomeType.SAVANNA
                        
    def _form_wetlands(self):
        """Form wetlands in appropriate locations."""
        if not hasattr(self.graph, 'water_flux'):
            return
            
        # Find areas with high water accumulation but not rivers
        for i in range(len(self.graph.points)):
            if (self.graph.heights[i] >= 20 and  # Land
                self.biomes[i] != BiomeType.RIVER and
                hasattr(self.graph, 'water_flux')):
                
                # High water flux indicates potential wetland
                water_flux = self.graph.water_flux[i]
                max_flux = np.max(self.graph.water_flux)
                
                if water_flux > max_flux * self.options.wetland_threshold:
                    # Check if it's not already a major river
                    is_major_river = False
                    if hasattr(self.graph, 'rivers'):
                        for river in self.graph.rivers:
                            if i in river.cells and river.flow > 100:
                                is_major_river = True
                                break
                                
                    if not is_major_river:
                        self.biomes[i] = BiomeType.WETLAND
                        
    def _form_mangroves(self):
        """Form mangroves in tropical coastal areas."""
        if not hasattr(self.graph, 'distance_field') or self.graph.distance_field is None:
            return
            
        for i in range(len(self.graph.points)):
            if (self.graph.heights[i] >= 20 and  # Land
                abs(self.graph.distance_field[i]) <= 1.0 and  # Very close to coast
                self.graph.temperatures[i] > 20):  # Tropical temperature
                
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
                area = len(region_cells) * (self.graph.spacing ** 2)
                
                region = BiomeRegion(
                    id=region_id,
                    biome_type=biome_type,
                    cells=region_cells,
                    area=area,
                    center_cell=center_cell
                )
                
                self.biome_regions.append(region)
                processed.update(region_cells)
                region_id += 1
                
        # Store on graph
        self.graph.biome_regions = self.biome_regions
        
        logger.info("Biome regions generated", count=len(self.biome_regions))
        
    def _find_connected_biome_cells(self, start_cell: int, biome_type: BiomeType, 
                                    processed: Set[int]) -> Set[int]:
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
                    if (neighbor not in region_cells and 
                        neighbor not in processed and
                        BiomeType(self.biomes[neighbor]) == biome_type):
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
        min_distance = float('inf')
        center_cell = next(iter(region_cells))
        
        for cell in region_cells:
            x, y = self.graph.points[cell]
            distance = (x - centroid_x)**2 + (y - centroid_y)**2
            
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
        
        logger.info("Biome classification completed",
                   biomes=len(np.unique(self.biomes)),
                   regions=len(self.biome_regions))
        
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

=======
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
