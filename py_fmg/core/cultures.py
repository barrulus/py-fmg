"""
Culture generation system based on FMG's culture-generators.js.

This module implements cultural region generation with different culture types,
expansion patterns, and geographical preferences matching the original FMG.
"""

from __future__ import annotations
import numpy as np
import structlog
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict
from .alea_prng import AleaPRNG
from .biomes import BiomeClassifier
from .voronoi_graph import (
    VoronoiGraph,
    build_cell_connectivity,
    build_cell_vertices,
    build_vertex_connectivity,
)

logger = structlog.get_logger()


class CultureOptions(BaseModel):
    """Culture generation options matching FMG's parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cultures_number: int = Field(default=12, description="Target number of cultures")
    min_culture_cells: int = Field(
        default=10, description="Minimum cells for valid culture"
    )
    expansionism_modifier: float = Field(
        default=1.0, description="Global expansionism multiplier"
    )

    # Culture type distribution weights
    generic_weight: float = Field(
        default=10.0, description="Weight for generic cultures"
    )
    naval_weight: float = Field(default=2.0, description="Weight for naval cultures")
    nomadic_weight: float = Field(
        default=1.0, description="Weight for nomadic cultures"
    )
    hunting_weight: float = Field(
        default=1.0, description="Weight for hunting cultures"
    )
    highland_weight: float = Field(
        default=1.0, description="Weight for highland cultures"
    )
    lake_weight: float = Field(default=0.5, description="Weight for lake cultures")
    river_weight: float = Field(default=1.0, description="Weight for river cultures")


class Culture(BaseModel):
    """Data structure for a cultural group."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(description="Unique culture identifier")
    name: str = Field(description="Culture name")
    color: str = Field(description="Culture color in hex format")
    center: int = Field(description="Cell ID of culture center")
    type: str = Field(default="Generic", description="Culture type")
    expansionism: float = Field(default=1.0, description="Expansion tendency")
    cells: Set[int] = Field(
        default_factory=set, description="Cells belonging to this culture"
    )
    removed: bool = Field(default=False, description="Whether culture has been removed")
    name_base: int = Field(default=0, description="Index into name bases")


class CultureGenerator:
    """Generates cultural regions based on FMG's algorithm."""

    def __init__(
        self,
        graph: VoronoiGraph,
        features: Any,
        biome_classifier: Optional[BiomeClassifier] = None,
        options: Optional[CultureOptions] = None,
        prng: Optional[AleaPRNG] = None,
    ):
        """
        Initialize culture generator.

        Args:
            graph: VoronoiGraph with populated data
            features: Features instance with geographic data
            biome_classifier: BiomeClassifier for habitability calculation
            options: Culture generation options
            prng: Random number generator for reproducible results
        """
        self.graph = graph
        self.features = features
        self.biome_classifier = biome_classifier or BiomeClassifier(self.graph) 
        self.options = options or CultureOptions()
        self.prng = prng or AleaPRNG("cultures")

        # Culture data
        self.cultures: Dict[int, Culture] = {}
        self.cell_cultures = np.zeros(len(graph.points), dtype=np.int32)
        self.next_culture_id = 1

        # Population data (moved from settlements as per FMG)
        self.cell_population = np.zeros(len(graph.points), dtype=np.float32)
        self.cell_suitability = np.zeros(len(graph.points), dtype=np.int16)

        # Color palette for cultures (from FMG)
        self.culture_colors = [
            "#9e2a2b",
            "#e55934",
            "#f17c67",
            "#a53253",
            "#ce4a81",
            "#d4a259",
            "#c9850d",
            "#e8b511",
            "#6ba9cb",
            "#4682b4",
            "#0f8040",
            "#1a4b5c",
            "#8b4513",
            "#daa520",
            "#ff6347",
            "#4169e1",
            "#32cd32",
            "#ff1493",
            "#00ced1",
            "#ffd700",
        ]

    def generate(self) -> Tuple[Dict[int, Culture], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete culture system.

        Returns:
            Tuple of (cultures dict, cell_cultures array)
        """
        logger.info("Starting culture generation")

        # Step 1: Calculate population and suitability (as per FMG)
        self._calculate_population()

        # Step 2: Place culture seed points
        seeds = self._place_culture_seeds()

        # Step 3: Create initial cultures from seeds
        self._create_cultures_from_seeds(seeds)

        # Step 4: Expand cultures across the map
        self._expand_cultures()

        # Step 5: Determine culture types based on geography
        self._determine_culture_types()

        # Step 6: Assign name bases to cultures
        self._assign_name_bases()

        # Step 7: Remove cultures that are too small
        self._cleanup_small_cultures()
        logger.info(f"Generated {len(self.cultures)} cultures")
        return (
            self.cultures,
            self.cell_cultures,
            self.cell_population,
            self.cell_suitability,
        )

    def _calculate_population(self) -> None:
        """
        Calculate cell population and suitability scores.

        This is moved from settlements as per FMG architecture where population
        is calculated during culture generation.
        """
        logger.info("Calculating population and suitability")

        # Get flux statistics for normalization
        flux_values = (
            self.graph.flux
            if self.graph.flux is not None
            else np.zeros(len(self.graph.points))
        )
        confluence_values = (
            self.graph.confluences
            if self.graph.confluences is not None
            else np.zeros(len(self.graph.points))
        )

        # Calculate mean and max flux for normalization
        land_flux = flux_values[self.graph.heights >= 20]
        fl_mean = np.median(land_flux[land_flux > 0]) if len(land_flux) > 0 else 0
        fl_max = (
            np.max(flux_values) + np.max(confluence_values)
            if len(flux_values) > 0
            else 1
        )

        # Area normalization
        area_mean = (
            np.mean(self.graph.cell_areas) if self.graph.cell_areas is not None else 1.0
        )

        for i in range(len(self.graph.points)):
            # Skip water cells
            if self.graph.heights[i] < 20:
                continue

            # Get biome habitability as base suitability
            habitability = self._get_cell_habitability(i)

            if habitability == 0:
                continue  # Uninhabitable areas

            s = float(habitability)

            # Rivers and confluences are highly valued
            if fl_mean > 0 and i < len(flux_values):
                flux_score = (
                    self._normalize(
                        flux_values[i] + confluence_values[i], fl_mean, fl_max
                    )
                    * 250
                )
                s += flux_score

            # Low elevation is valued, high is not
            s -= (self.graph.heights[i] - 50) / 5

            # Coastal and lake shores get bonuses
            if self.graph.cell_types is not None and i < len(self.graph.cell_types):
                if self.graph.cell_types[i] == 1:  # Coastline
                    if self.graph.river_ids is not None and self.graph.river_ids[i] > 0:
                        s += 15  # Estuary bonus

                    # Check if it's a lake shore
                    if self.graph.cell_haven is not None and i < len(
                        self.graph.cell_haven
                    ):
                        haven = self.graph.cell_haven[i]
                        if haven > 0 and haven < len(self.features.features):
                            feature = self.features.features[haven]
                            if hasattr(feature, "type") and feature.type == "lake":
                                if (
                                    hasattr(feature, "freshwater")
                                    and feature.freshwater
                                ):
                                    s += 30  # Freshwater lake bonus
                                else:
                                    s += 10  # Salt lake bonus
                            elif hasattr(feature, "type") and feature.type == "ocean":
                                s += 25  # Ocean access bonus
                else:
                    s -= 5  # Non-coastal penalty

            # Store suitability score (clamped to int16 range)
            self.cell_suitability[i] = max(0, min(int(s), 32767))

            # Calculate population based on suitability and area
            if self.graph.cell_areas is not None and i < len(self.graph.cell_areas):
                area_factor = self.graph.cell_areas[i] / area_mean
            else:
                area_factor = 1.0

            self.cell_population[i] = max(0, s * area_factor / 100)

    def _normalize(self, value: float, mean: float, max_val: float) -> float:
        """Normalize value using FMG's normalization formula."""
        if max_val == 0:
            return 0
        return min(1, value / mean) * (1 - mean / max_val)

    def _place_culture_seeds(self) -> List[int]:
        """
        Place culture seed points using FMG's algorithm.

        Returns:
            List of cell IDs for culture centers
        """
        logger.info("Placing culture seeds")

        # Get all land cells with population potential (now pre-calculated)
        valid_cells = []
        for i in range(len(self.graph.points)):
            if (
                self.graph.heights[i] >= 20 and self.cell_population[i] > 0  # Land cell
            ):  # Has population potential
                valid_cells.append(i)

        if len(valid_cells) < self.options.cultures_number:
            logger.warning(
                f"Not enough valid cells for {self.options.cultures_number} cultures"
            )
            self.options.cultures_number = max(1, len(valid_cells) // 10)

        # Calculate minimum distance between seeds
        map_size = math.sqrt(self.graph.graph_width * self.graph.graph_height)
        min_distance = map_size / (self.options.cultures_number**0.7)

        seeds: List[int] = []
        attempts = 0
        max_attempts = len(valid_cells) * 2

        while len(seeds) < self.options.cultures_number and attempts < max_attempts:
            attempts += 1

            # Pick random cell
            cell_id = self.prng.choice(valid_cells)

            if not seeds:
                seeds.append(cell_id)
                continue

            # Check distance to existing seeds
            x, y = self.graph.points[cell_id]
            too_close = False

            for seed_id in seeds:
                sx, sy = self.graph.points[seed_id]
                dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                seeds.append(cell_id)
            elif attempts > max_attempts // 2:
                # Reduce distance requirement if struggling
                min_distance *= 0.9

        logger.info(f"Placed {len(seeds)} culture seeds")
        return seeds

    def _create_cultures_from_seeds(self, seeds: List[int]) -> None:
        """Create initial culture objects from seed points."""
        logger.info("Creating cultures from seeds")

        for i, seed_cell in enumerate(seeds):
            culture_id = self.next_culture_id

            # Assign expansionism factor (0.8 to 1.2)
            expansionism = 0.8 + self.prng.random() * 0.4
            expansionism *= self.options.expansionism_modifier

            culture = Culture(
                id=culture_id,
                name=f"Culture {culture_id}",  # Will be replaced with proper names
                color=self.culture_colors[i % len(self.culture_colors)],
                center=seed_cell,
                expansionism=expansionism,
            )

            self.cultures[culture_id] = culture
            self.cell_cultures[seed_cell] = culture_id
            culture.cells.add(seed_cell)
            self.next_culture_id += 1

    def _expand_cultures(self) -> None:
        """
        Expand cultures across the map using cost-based algorithm.

        This implements FMG's culture expansion similar to state expansion
        but focused on geographic and population factors.
        """
        logger.info("Expanding cultures")

        import heapq

        # Priority queue: (cost, cell_id, culture_id)
        heap: List[Tuple[float, int, int]] = []
        costs: Dict[int, float] = {}

        # Initialize with culture centers
        for culture in self.cultures.values():
            heapq.heappush(heap, (0, culture.center, culture.id))
            costs[culture.center] = 0

        # Expansion parameters
        max_expansion_cost = len(self.graph.points) * 0.5

        while heap:
            current_cost, current_cell, culture_id = heapq.heappop(heap)

            if current_cost > max_expansion_cost:
                continue

            culture = self.cultures[culture_id]

            # Get neighbors
            neighbors = (
                self.graph.cell_neighbors[current_cell]
                if current_cell < len(self.graph.cell_neighbors)
                else []
            )

            for neighbor in neighbors:
                # Skip if already assigned to a culture
                if self.cell_cultures[neighbor] != 0:
                    continue

                # Skip water cells
                if self.graph.heights[neighbor] < 20:
                    continue

                # Calculate expansion cost
                expansion_cost = self._calculate_culture_expansion_cost(
                    neighbor, culture_id, current_cell
                )

                total_cost = current_cost + 10 + expansion_cost / culture.expansionism

                if total_cost > max_expansion_cost:
                    continue

                # Update if this is a better path
                if neighbor not in costs or total_cost < costs[neighbor]:
                    self.cell_cultures[neighbor] = culture_id
                    culture.cells.add(neighbor)
                    costs[neighbor] = total_cost
                    heapq.heappush(heap, (total_cost, neighbor, culture_id))

    def _calculate_culture_expansion_cost(
        self, cell_id: int, culture_id: int, from_cell: int
    ) -> float:
        """Calculate cost for culture to expand into a cell."""
        cost = 0.0

        # Population preference - cultures like populated areas
        pop = self.cell_population[cell_id]
        if pop > 0:
            cost -= min(pop * 5, 30)  # Bonus for population
        else:
            cost += 20  # Penalty for unpopulated areas

        # Height cost - avoid extreme elevations
        height = self.graph.heights[cell_id]
        if height > 70:  # Mountains
            cost += 50
        elif height > 50:  # Hills
            cost += 10

        # Distance from culture center
        culture = self.cultures[culture_id]
        cx, cy = self.graph.points[culture.center]
        x, y = self.graph.points[cell_id]
        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        cost += distance / 50  # Small distance penalty

        # Coastal preference for some expansions
        if self.graph.cell_types is not None and cell_id < len(self.graph.cell_types):
            if self.graph.cell_types[cell_id] == 1:  # Coastline
                # Random cultures get coastal bonus
                if self.prng.random() < 0.3:
                    cost -= 15

        return max(cost, 0)

    def _determine_culture_types(self) -> None:
        """Determine culture types based on geographic location and preferences."""
        logger.info("Determining culture types")

        for culture in self.cultures.values():
            culture.type = self._get_culture_type_for_center(culture.center, culture.id)

    def _get_culture_type_for_center(self, center_cell: int, culture_id: int) -> str:
        """Determine culture type based on geographic features around center."""

        # Count different terrain types in culture territory
        culture = self.cultures[culture_id]
        coastal_cells = 0
        highland_cells = 0
        river_cells = 0
        lake_cells = 0
        total_cells = len(culture.cells)

        for cell_id in culture.cells:
            height = self.graph.heights[cell_id]

            # Highland check
            if height > 60:
                highland_cells += 1

            # Coastal check
            if (
                self.graph.cell_types is not None
                and cell_id < len(self.graph.cell_types)
                and self.graph.cell_types[cell_id] == 1
            ):
                coastal_cells += 1

            # River check
            if (
                self.graph.river_ids is not None
                and cell_id < len(self.graph.river_ids)
                and self.graph.river_ids[cell_id] > 0
            ):
                river_cells += 1

            # Lake check
            if self.graph.cell_haven is not None and cell_id < len(
                self.graph.cell_haven
            ):
                haven = self.graph.cell_haven[cell_id]
                if haven > 0 and haven < len(self.features.features):
                    feature = self.features.features[haven]
                    if hasattr(feature, "type") and feature.type == "lake":
                        lake_cells += 1

        # Calculate ratios
        coastal_ratio = coastal_cells / total_cells if total_cells > 0 else 0
        highland_ratio = highland_cells / total_cells if total_cells > 0 else 0
        river_ratio = river_cells / total_cells if total_cells > 0 else 0
        lake_ratio = lake_cells / total_cells if total_cells > 0 else 0

        # Determine type based on dominant terrain
        if coastal_ratio > 0.4:
            return "Naval"
        elif highland_ratio > 0.5:
            return "Highland"
        elif lake_ratio > 0.3:
            return "Lake"
        elif river_ratio > 0.4:
            return "River"
        elif sum(self.cell_population[c] for c in culture.cells) / total_cells < 2:
            return "Nomadic"
        else:
            return "Generic"

    def _assign_name_bases(self) -> None:
        """Assign name bases to cultures based on geography and culture type."""
        logger.info("Assigning name bases to cultures")

        # Regional name base assignments based on geography and culture type
        # This creates diverse cultural naming patterns across the map

        # Define name base groups by cultural affinity
        continental_bases = [0, 2, 3]  # German, French, Italian (inland cultures)
        maritime_bases = [1, 6, 7]  # English, Nordic, Greek (coastal cultures)
        nomadic_bases = [4, 5]  # Spanish, Ruthenian (nomadic/trade cultures)
        highland_bases = [0, 6, 8]  # German, Nordic, Roman (mountain cultures)

        # Calculate map center for regional assignments
        center_x = self.graph.graph_width / 2
        center_y = self.graph.graph_height / 2

        for culture in self.cultures.values():
            # Get culture center position
            cx, cy = self.graph.points[culture.center]

            # Calculate position relative to map center
            rel_x = (cx - center_x) / center_x  # -1 to 1
            rel_y = (cy - center_y) / center_y  # -1 to 1

            # Base name base selection on culture type and position
            if culture.type == "Naval":
                # Naval cultures prefer maritime name bases
                if abs(rel_x) > abs(rel_y):  # East-West preference
                    culture.name_base = maritime_bases[0 if rel_x < 0 else 1]
                else:
                    culture.name_base = maritime_bases[2]

            elif culture.type == "Highland":
                # Highland cultures use mountain-associated bases
                # North = Nordic, South = Roman, Central = German
                if rel_y < -0.3:  # Northern highlands
                    culture.name_base = highland_bases[1]  # Nordic
                elif rel_y > 0.3:  # Southern highlands
                    culture.name_base = highland_bases[2]  # Roman
                else:
                    culture.name_base = highland_bases[0]  # German

            elif culture.type == "River":
                # River cultures prefer continental bases with good flow
                culture.name_base = continental_bases[
                    hash(culture.id) % len(continental_bases)
                ]

            elif culture.type == "Lake":
                # Lake cultures use peaceful continental names
                culture.name_base = continental_bases[1]  # French (most lake-like)

            elif culture.type == "Nomadic":
                # Nomadic cultures use distinctive bases
                culture.name_base = nomadic_bases[culture.id % len(nomadic_bases)]

            else:  # Generic cultures
                # Distribute generic cultures geographically
                if abs(rel_x) + abs(rel_y) < 0.5:  # Central region
                    culture.name_base = continental_bases[1]  # French
                elif rel_x < -0.3:  # Western region
                    culture.name_base = continental_bases[0]  # German
                elif rel_x > 0.3:  # Eastern region
                    culture.name_base = continental_bases[2]  # Italian
                elif rel_y < 0:  # Northern region
                    culture.name_base = maritime_bases[1]  # Nordic
                else:  # Southern region
                    culture.name_base = maritime_bases[2]  # Greek

            # Ensure name base is within valid range (fallback to cycling)
            if culture.name_base >= 12:  # Assuming 12 bases available
                culture.name_base = culture.id % 12

    def _cleanup_small_cultures(self) -> None:
        """Remove cultures that are too small and reassign their cells."""
        logger.info("Cleaning up small cultures")

        cultures_to_remove = []
        for culture in self.cultures.values():
            if len(culture.cells) < self.options.min_culture_cells:
                cultures_to_remove.append(culture.id)

        for culture_id in cultures_to_remove:
            culture = self.cultures[culture_id]
            culture.removed = True

            # Reassign cells to neighboring cultures
            for cell_id in culture.cells:
                self.cell_cultures[cell_id] = 0  # Mark as unassigned

                # Find best neighboring culture
                neighbors = (
                    self.graph.cell_neighbors[cell_id]
                    if cell_id < len(self.graph.cell_neighbors)
                    else []
                )

                best_culture = 0
                for neighbor in neighbors:
                    neighbor_culture = self.cell_cultures[neighbor]
                    if (
                        neighbor_culture > 0
                        and not self.cultures[neighbor_culture].removed
                    ):
                        best_culture = neighbor_culture
                        break

                if best_culture > 0:
                    self.cell_cultures[cell_id] = best_culture
                    self.cultures[best_culture].cells.add(cell_id)

            # Remove culture
            del self.cultures[culture_id]

        if cultures_to_remove:
            logger.info(f"Removed {len(cultures_to_remove)} small cultures")

    def _get_cell_habitability(self, cell_id: int) -> int:
        """
        Get habitability score for a cell using biome classification.

        Replaces the old height-based system with proper biome integration
        for realistic population distribution patterns.

        Args:
            cell_id: Cell index

        Returns:
            Habitability score (0-100, where 100 is most habitable)
        """
        # Check if we have biome data available
        if self.graph.biomes is not None and hasattr(self.graph.biomes, "cell_biomes"):
            # Use actual biome classification
            if cell_id < len(self.graph.biomes):
                biome_id = self.graph.biomes[cell_id]
                return int(self.biome_classifier.get_biome_habitability(biome_id))

        # Fallback: Calculate biome from climate data if available
        if (
            self.graph.temperatures is not None
            and self.graph.precipitation is not None
            and cell_id < len(self.graph.temperatures)
            and cell_id < len(self.graph.precipitation)
        ):

            temperature = self.graph.temperatures[cell_id]
            precipitation = self.graph.precipitation[cell_id]
            height = self.graph.heights[cell_id]

            # Calculate moisture (simplified from biome classifier)
            moisture = 4.0 + precipitation
            if self.graph.river_ids is not None and cell_id < len(self.graph.river_ids):
                if self.graph.river_ids[cell_id] > 0:
                    moisture += 2.0  # River bonus

        # Final fallback: Enhanced height-based calculation with climate considerations
        height = self.graph.heights[cell_id]

        # Base habitability from height
        if height > 80:
            base_habitability = 10  # High mountains - very low habitability
        elif height > 60:
            base_habitability = 35  # Hills - moderate habitability
        else:
            base_habitability = 80  # Lowlands - high habitability

        # Climate adjustments if available
        if self.graph.temperatures is not None and cell_id < len(
            self.graph.temperatures
        ):
            temperature = self.graph.temperatures[cell_id]

            # Temperature penalties
            if temperature < -10:
                base_habitability = int(base_habitability * 0.2)  # Very cold
            elif temperature < -5:
                base_habitability = int(base_habitability * 0.4)  # Cold
            elif temperature > 35:
                base_habitability = int(base_habitability * 0.3)  # Very hot
            elif temperature > 30:
                base_habitability = int(base_habitability * 0.6)  # Hot
        # Precipitation adjustments if available
        if self.graph.precipitation is not None and cell_id < len(
            self.graph.precipitation
        ):
            precipitation = self.graph.precipitation[cell_id]

            # Very dry or very wet areas are less habitable
            if precipitation < 5:
                base_habitability = int(base_habitability * 0.3)  # Desert conditions
            elif precipitation < 10:
                base_habitability = int(base_habitability * 0.6)  # Dry conditions
            elif precipitation > 50:
                base_habitability = int(base_habitability * 0.7)  # Very wet conditions

        return int(max(0, min(100, base_habitability)))
