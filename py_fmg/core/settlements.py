"""
Settlement and state generation system.

This module handles settlement placement (capitals and towns) and state formation
following the original Fantasy Map Generator algorithms.

Process:
1. rankCells() - Calculate cell suitability scores
2. placeCapitals() - Place capital cities with spacing constraints
3. createStates() - Initialize states from capitals
4. placeTowns() - Add secondary settlements
5. expandStates() - Territorial expansion using cost-based algorithm
6. normalizeStates() - Clean up state boundaries
"""

import heapq

from dataclasses import dataclass, field

from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pydantic import BaseModel, Field, ConfigDict
from sklearn.neighbors import KDTree

from .name_generator import NameGenerator
from .voronoi_graph import (
    VoronoiGraph,
    build_cell_connectivity,
    build_cell_vertices,
    build_vertex_connectivity,
)

logger = structlog.get_logger()


class SettlementOptions(BaseModel):
    """Settlement generation options matching FMG's parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    states_number: int = Field(default=30, description="Target number of states")
    manors_number: int = Field(
        default=1000, description="Target number of towns (1000 = auto)"
    )
    growth_rate: float = Field(default=1.0, description="Global growth rate multiplier")
    states_growth_rate: float = Field(
        default=1.0, description="States-specific growth multiplier"
    )
    size_variety: float = Field(
        default=3.0, description="Variation in state expansionism"
    )
    min_state_cells: int = Field(
        default=10, description="Minimum cells for valid state"
    )

    # Spacing parameters
    capital_spacing_divisor: int = Field(
        default=2, description="Divide map size by state count * this"
    )
    town_spacing_base: int = Field(
        default=150, description="Base divisor for town spacing"
    )
    town_spacing_power: float = Field(
        default=0.7, description="Power adjustment for town count"
    )

    # Cost parameters
    culture_same_bonus: int = Field(
        default=-9, description="Bonus for expanding into same culture"
    )
    culture_foreign_penalty: int = Field(
        default=100, description="Penalty for expanding into foreign culture"
    )
    sea_crossing_penalty: int = Field(
        default=1000, description="General sea crossing penalty"
    )
    nomadic_sea_penalty: int = Field(
        default=10000, description="Extreme penalty for nomadic sea crossing"
    )

    # Population parameters
    urbanization_rate: float = Field(
        default=0.1, description="Target ~10% urbanization"
    )
    capital_pop_multiplier: float = Field(
        default=1.3, description="Capital population boost"
    )
    port_pop_multiplier: float = Field(default=1.3, description="Port population boost")


class Settlement(BaseModel):
    """Data structure for a settlement (burg)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(description="Unique settlement identifier")
    cell_id: int = Field(description="Cell ID where settlement is located")
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    name: str = Field(default="", description="Settlement name")
    population: float = Field(default=0.0, description="Settlement population")
    is_capital: bool = Field(
        default=False, description="Whether this is a capital city"
    )
    state_id: int = Field(default=0, description="State ID this settlement belongs to")
    culture_id: int = Field(default=0, description="Culture ID of this settlement")
    port_id: int = Field(default=0, description="Feature ID if port, 0 otherwise")
    religion_id: int = Field(default=0, description="Religion ID if assigned")
    type: str = Field(default="Generic", description="Settlement type")

    # Features
    citadel: bool = Field(default=False, description="Has citadel")
    plaza: bool = Field(default=False, description="Has plaza")
    walls: bool = Field(default=False, description="Has walls")
    shanty: bool = Field(default=False, description="Has shanty town")
    temple: bool = Field(default=False, description="Has temple")


class State(BaseModel):
    """Data structure for a political state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(description="Unique state identifier")
    name: str = Field(description="State name")
    capital_id: int = Field(description="ID of capital settlement")
    culture_id: int = Field(description="Dominant culture ID")
    expansionism: float = Field(default=1.0, description="Expansion tendency")
    color: str = Field(default="#000000", description="State color in hex format")
    type: str = Field(
        default="Generic", description="Cultural type affecting expansion"
    )
    center_cell: int = Field(default=0, description="Center cell ID")
    territory_cells: List[int] = Field(
        default_factory=list, description="Cells controlled by this state"
    )
    removed: bool = Field(default=False, description="Whether state has been removed")
    locked: bool = Field(default=False, description="Whether state expansion is locked")


class Settlements:
    """Handles settlement placement and state generation."""

    def __init__(
        self,
        graph:VoronoiGraph,
        features,
        cultures,
        biomes,
        name_generator: NameGenerator,
        options: Optional[SettlementOptions] = None,
        cell_religions: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize Settlements with graph, features, cultures, biomes, name generator, and religions.

        Args:
            graph: VoronoiGraph instance with populated data
            features: Features instance with detected geographic features
            cultures: Cultures instance with cultural regions
            biomes: Biomes instance with biome classifications
            name_generator: NameGenerator for generating settlement and state names
            options: SettlementOptions for configuration
            cell_religions: Optional array mapping cell IDs to religion IDs
        """
        self.graph = graph
        self.features = features
        self.cultures = cultures
        self.biomes = biomes
        self.name_generator = name_generator
        self.options = options or SettlementOptions()

        # Religion integration
        self.cell_religions = (
            cell_religions
            if cell_religions is not None
            else np.zeros(len(graph.points), dtype=np.uint16)
        )

        # Initialize arrays
        # Population and suitability are now calculated in cultures module
        self.cell_suitability = getattr(
            graph, "cell_suitability", np.zeros(len(graph.points), dtype=np.int16)
        )
        self.cell_population = getattr(
            graph, "cell_population", np.zeros(len(graph.points), dtype=np.float32)
        )
        self.cell_state = np.zeros(len(graph.points), dtype=np.uint16)
        self.cell_settlement = np.zeros(len(graph.points), dtype=np.uint16)

        # Settlement and state data
        self.settlements: Dict[int, Settlement] = {}
        self.states: Dict[int, State] = {}
        self.next_settlement_id = 1
        self.next_state_id = 1

    def generate(self) -> Tuple[Dict[int, Settlement], Dict[int, State]]:
        """
        Generate complete settlement and state system.

        Returns:
            Tuple of (settlements dict, states dict)
        """
        logger.info("Starting settlement and state generation")

        # Step 1: Population and suitability are now pre-calculated in cultures
        # No need to rank cells again

        # Step 2: Place capital cities
        capitals = self.place_capitals()

        # Step 3: Create initial states from capitals
        self.create_states(capitals)

        # Step 4: Place secondary settlements (towns)
        self.place_towns()

        # Step 5: Expand states territorially
        self.expand_states()

        # Step 6: Normalize state boundaries
        self.normalize_states()

        # Step 7: Specify settlement details
        self.specify_settlements()

        logger.info(
            f"Generated {len(self.settlements)} settlements and {len(self.states)} states"
        )
        return self.settlements, self.states

    def rank_cells(self) -> None:
        """
        Calculate cell suitability scores for settlement placement.

        NOTE: This method is deprecated - population and suitability are now
        calculated in the cultures module as per FMG architecture.
        """
        logger.warning(
            "rank_cells() called but population is now calculated in cultures module"
        )
        
        # Get flux statistics for normalization using tile_events system
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

        # Area normalization using tile_events system
        area_mean = (
            np.mean(self.graph.cell_areas) if self.graph.cell_areas is not None else 1.0
        )

        for i in range(len(self.graph.points)):
            # Skip water cells
            if self.graph.heights[i] < 20:
                continue

            # Get biome habitability as base suitability
            biome_id = (
                self.biomes.cell_biomes[i] if hasattr(self.biomes, "cell_biomes") else 1
            )
            habitability = (
                self.biomes.get_habitability(biome_id)
                if hasattr(self.biomes, "get_habitability")
                else 100
            )

            if habitability == 0:
                continue  # Uninhabitable biomes

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
            return 

    def _normalize(self, value: float, mean: float, max_val: float) -> float:
        """Normalize value using FMG's normalization formula."""
        if max_val == 0:
            return 0
        return min(1, value / mean) * (1 - mean / max_val)

    def place_capitals(self) -> List[Settlement]:
        """
        Place capital cities with minimum spacing constraints.

        Returns:
            List of placed capital settlements
        """
        logger.info("Placing capital cities")

        count = self.options.states_number
        capitals = []

        # Random factor for score variation (0.5 to 1.0)
        rand_factors = 0.5 + np.random.random(len(self.cell_suitability)) * 0.5

        # Calculate scores with random variation
        scores = self.cell_suitability * rand_factors

        # Filter for populated cells with positive scores
        valid_cells = []
        for i in range(len(scores)):
            if (
                scores[i] > 0
                and self.graph.heights[i] >= 20
                and hasattr(self.cultures, "cell_cultures")
                and self.cultures.cell_cultures[i] > 0
            ):
                valid_cells.append((scores[i], i))

        # Sort by score (highest first)
        valid_cells.sort(reverse=True, key=lambda x: x[0])

        if len(valid_cells) < count * 10:
            # Adjust count if not enough valid cells
            count = max(1, len(valid_cells) // 10)
            logger.warning(f"Not enough populated cells. Reducing states to {count}")

        # Calculate initial spacing
        map_size = (self.graph.graph_width + self.graph.graph_height) / 2
        spacing = map_size / count / self.options.capital_spacing_divisor

        # Use KDTree for efficient spatial queries
        placed_positions = []

        attempts = 0
        max_attempts = 3

        while len(capitals) < count and attempts < max_attempts:
            for score, cell_id in valid_cells:
                if len(capitals) >= count:
                    break

                x, y = self.graph.points[cell_id]

                # Check spacing constraint
                if placed_positions:
                    tree = KDTree(placed_positions)
                    distances, _ = tree.query([[x, y]], k=1)
                    if distances[0][0] < spacing:
                        continue

                # Create capital settlement
                capital = Settlement(
                    id=self.next_settlement_id,
                    cell_id=cell_id,
                    x=x,
                    y=y,
                    is_capital=True,
                    culture_id=(
                        self.cultures.cell_cultures[cell_id]
                        if hasattr(self.cultures, "cell_cultures")
                        else 0
                    ),
                    religion_id=self._get_religion_for_cell(cell_id),
                )

                capitals.append(capital)
                placed_positions.append([x, y])
                self.settlements[capital.id] = capital
                self.cell_settlement[cell_id] = capital.id
                self.next_settlement_id += 1

            if len(capitals) < count:
                # Reduce spacing and try again
                spacing /= 1.2
                attempts += 1
                logger.warning(
                    f"Retrying capital placement with reduced spacing: {spacing:.2f}"
                )

        logger.info(f"Placed {len(capitals)} capital cities")
        return capitals

    def create_states(self, capitals: List[Settlement]) -> None:
        """Create initial states from placed capitals."""
        logger.info("Creating states from capitals")

        # Create neutral state
        self.states[0] = State(
            id=0, name="Neutrals", capital_id=0, culture_id=0, color="#808080"
        )

        for capital in capitals:
            state_id = self.next_state_id

            # Assign expansionism factor
            expansionism = np.random.random() * self.options.size_variety + 1.0

            # Create state
            state = State(
                id=state_id,
                name=self.name_generator.generate_culture_short(capital.culture_id),
                capital_id=capital.id,
                culture_id=capital.culture_id,
                expansionism=round(expansionism, 1),
                center_cell=capital.cell_id,
                type=self._get_culture_type(capital.culture_id),
            )

            self.states[state_id] = state
            capital.state_id = state_id
            self.cell_state[capital.cell_id] = state_id
            self.next_state_id += 1

    def _get_culture_type(self, culture_id: int) -> str:
        """Get culture type that affects state expansion."""
        if hasattr(self.cultures, "cultures") and culture_id in self.cultures.cultures:
            return self.cultures.cultures[culture_id].type
        return "Generic"

    def place_towns(self) -> None:
        """Place secondary settlements based on suitability scores."""
        logger.info("Placing towns")

        # Calculate scores with more variation for towns
        gauss_factors = (
            np.clip(np.random.normal(1, 3, len(self.cell_suitability)), 0, 20) / 3
        )
        scores = self.cell_suitability * gauss_factors

        # Filter for valid town locations (not already occupied)
        valid_cells = []
        for i in range(len(scores)):
            if (
                self.cell_settlement[i] == 0
                and scores[i] > 0
                and self.graph.heights[i] >= 20
                and hasattr(self.cultures, "cell_cultures")
                and self.cultures.cell_cultures[i] > 0
            ):
                valid_cells.append((scores[i], i))

        # Sort by score
        valid_cells.sort(reverse=True, key=lambda x: x[0])

        # Determine number of towns
        if self.options.manors_number == 1000:  # Auto mode
            # Scale based on map size
            grid_size_factor = (len(self.graph.points) / 10000) ** 0.8
            desired_number = int(len(valid_cells) / 5 / grid_size_factor)
        else:
            desired_number = self.options.manors_number

        towns_number = min(desired_number, len(valid_cells))

        # Calculate spacing (avoid division by zero)
        spacing = (
            self.graph.graph_width + self.graph.graph_height
        ) / self.options.town_spacing_base
        if towns_number > 0:
            spacing /= towns_number**self.options.town_spacing_power / 66
        else:
            # No towns to place, set a default spacing
            spacing = 1.0

        # Get existing settlement positions
        existing_positions = []
        for settlement in self.settlements.values():
            existing_positions.append([settlement.x, settlement.y])

        towns_added = 0

        while towns_added < towns_number and spacing > 1:
            for score, cell_id in valid_cells:
                if towns_added >= towns_number:
                    break

                if self.cell_settlement[cell_id] != 0:
                    continue

                x, y = self.graph.points[cell_id]

                # Randomize spacing for non-uniform placement
                s = spacing * np.clip(np.random.normal(1, 0.3), 0.2, 2)

                # Check spacing
                if existing_positions:
                    tree = KDTree(existing_positions)
                    distances, _ = tree.query([[x, y]], k=1)
                    if distances[0][0] < s:
                        continue

                # Create town
                town = Settlement(
                    id=self.next_settlement_id,
                    cell_id=cell_id,
                    x=x,
                    y=y,
                    is_capital=False,
                    culture_id=(
                        self.cultures.cell_cultures[cell_id]
                        if hasattr(self.cultures, "cell_cultures")
                        else 0
                    ),
                    state_id=0,  # Will be assigned during expansion
                    religion_id=self._get_religion_for_cell(cell_id),
                )

                self.settlements[town.id] = town
                self.cell_settlement[cell_id] = town.id
                existing_positions.append([x, y])
                self.next_settlement_id += 1
                towns_added += 1

            spacing *= 0.5

        logger.info(f"Placed {towns_added} towns")

    def expand_states(self) -> None:
        """
        Expand states territorially using cost-based algorithm.

        This is the most complex political algorithm, using a priority queue
        to grow states outward from capitals based on expansion costs.
        """
        logger.info("Expanding states territorially")

        # Clear non-locked state assignments
        for i in range(len(self.cell_state)):
            state = self.states.get(self.cell_state[i])
            if state and not state.locked:
                self.cell_state[i] = 0

        # Calculate growth rate limit
        growth_rate = (
            len(self.graph.points)
            / 2
            * self.options.growth_rate
            * self.options.states_growth_rate
        )

        # Priority queue: (cost, cell_id, state_id, native_biome)
        heap = []
        costs = {}

        # Initialize queue with state capitals
        for state_id, state in self.states.items():
            if state_id == 0 or state.removed:
                continue

            capital = self.settlements.get(state.capital_id)
            if not capital:
                continue

            capital_cell = capital.cell_id
            self.cell_state[capital_cell] = state_id

            # Get state's native biome from culture center
            native_biome = 1  # Default
            if (
                hasattr(self.cultures, "cultures")
                and state.culture_id in self.cultures.cultures
            ):
                culture = self.cultures.cultures[state.culture_id]
                if hasattr(culture, "center"):
                    center_cell = culture.center
                    if hasattr(self.biomes, "cell_biomes") and center_cell < len(
                        self.biomes.cell_biomes
                    ):
                        native_biome = self.biomes.cell_biomes[center_cell]

            heapq.heappush(heap, (0, state.center_cell, state_id, native_biome))
            costs[state.center_cell] = 0

        # Expand states
        while heap:
            current_cost, current_cell, state_id, native_biome = heapq.heappop(heap)

            if current_cost > growth_rate:
                continue

            state = self.states[state_id]

            # Check all neighbors
            neighbors = (
                self.graph.cell_neighbors[current_cell]
                if current_cell < len(self.graph.cell_neighbors)
                else []
            )

            for neighbor in neighbors:
                # Skip if neighbor state is locked
                neighbor_state_id = self.cell_state[neighbor]
                if neighbor_state_id > 0:
                    neighbor_state = self.states.get(neighbor_state_id)
                    if neighbor_state and neighbor_state.locked:
                        continue
                    # Don't overwrite other capitals
                    if neighbor == neighbor_state.center_cell:
                        continue

                # Calculate expansion cost
                cell_cost = self._calculate_expansion_cost(
                    neighbor, state_id, state.culture_id, state.type, native_biome
                )

                total_cost = current_cost + 10 + cell_cost / state.expansionism

                if total_cost > growth_rate:
                    continue

                # Update if this is a better path
                if neighbor not in costs or total_cost < costs[neighbor]:
                    if self.graph.heights[neighbor] >= 20:  # Only expand on land
                        self.cell_state[neighbor] = state_id
                    costs[neighbor] = total_cost
                    heapq.heappush(heap, (total_cost, neighbor, state_id, native_biome))

        # Assign states to settlements
        for settlement in self.settlements.values():
            if not settlement.is_capital:
                settlement.state_id = self.cell_state[settlement.cell_id]

    def _calculate_expansion_cost(
        self,
        cell_id: int,
        state_id: int,
        state_culture: int,
        state_type: str,
        native_biome: int,
    ) -> float:
        """Calculate cost for state to expand into a cell with specialized state type rules."""
        cost = 0

        # Culture affinity (enhanced for cultural diversity)
        cell_culture = (
            self.cultures.cell_cultures[cell_id]
            if hasattr(self.cultures, "cell_cultures")
            else 0
        )
        if cell_culture == state_culture:
            # Same culture bonus - stronger for certain state types
            culture_bonus = self.options.culture_same_bonus
            if state_type in ["Highland", "Nomadic"]:
                culture_bonus *= (
                    1.5  # Highland and nomadic cultures are more culturally cohesive
                )
            cost += culture_bonus
        else:
            # Foreign culture penalty - varies by state type
            foreign_penalty = self.options.culture_foreign_penalty
            if state_type == "Naval":
                foreign_penalty *= 0.7  # Naval states are more culturally tolerant
            elif state_type == "Highland":
                foreign_penalty *= 1.3  # Highland states are more insular
            cost += foreign_penalty

        # Population preference (state type specific)
        if self.graph.heights[cell_id] < 20:
            # Water cells - only naval states can efficiently expand here
            if state_type != "Naval":
                cost += 200  # High penalty for non-naval expansion over water
        elif self.cell_suitability[cell_id] > 0:
            pop_cost = max(20 - self.cell_suitability[cell_id], 0)
            # State type population preferences
            if (
                state_type == "River"
                and self.graph.river_ids is not None
                and self.graph.river_ids[cell_id] > 0
            ):
                pop_cost *= (
                    0.5  # River states prefer river cells even if less populated
                )
            cost += pop_cost
        else:
            # Unsuitable land penalty varies by state type
            unsuitable_penalty = 5000
            if state_type == "Nomadic":
                unsuitable_penalty *= 0.6  # Nomads can utilize marginal land better
            elif state_type == "Highland":
                unsuitable_penalty *= 0.8  # Highland peoples adapt to harsh terrain
            cost += unsuitable_penalty

        # Biome adaptation cost
        cell_biome = (
            self.biomes.cell_biomes[cell_id]
            if hasattr(self.biomes, "cell_biomes")
            else 1
        )
        cost += self._get_biome_cost(native_biome, cell_biome, state_type)

        # Terrain cost with state type specialization
        height = self.graph.heights[cell_id]
        cost += self._get_height_cost(height, state_type, cell_id)

        # River expansion preferences
        if self.graph.river_ids is not None and self.graph.river_ids[cell_id] > 0:
            river_cost = self._get_river_cost(cell_id, state_type)
            # Special river state bonuses
            if state_type == "River":
                river_cost = max(0, river_cost - 30)  # Major bonus for river expansion
            cost += river_cost

        # Coastal terrain preferences
        if self.graph.cell_types is not None:
            terrain_type = self.graph.cell_types[cell_id]
            terrain_cost = self._get_terrain_cost(terrain_type, state_type)
            # Enhanced coastal preferences for naval states
            if state_type == "Naval" and terrain_type == 1:  # Coastline
                terrain_cost = max(0, terrain_cost - 25)  # Extra coastal bonus
            cost += terrain_cost

        # Distance from cultural homeland (enhanced political boundaries)
        state = self.states[state_id]
        if hasattr(state, "center_cell"):
            center_x, center_y = self.graph.points[state.center_cell]
            cell_x, cell_y = self.graph.points[cell_id]
            distance = ((center_x - cell_x) ** 2 + (center_y - cell_y) ** 2) ** 0.5

            # Distance penalty varies by state type
            if state_type == "Nomadic":
                distance_penalty = distance / 100  # Nomads expand widely
            elif state_type == "Highland":
                distance_penalty = distance / 30  # Highland states stay compact
            elif state_type == "Naval":
                distance_penalty = (
                    distance / 80
                )  # Naval states can project power via sea
            else:
                distance_penalty = distance / 50  # Default distance penalty

            cost += distance_penalty

        return max(cost, 0)

    def _get_biome_cost(
        self, native_biome: int, cell_biome: int, state_type: str
    ) -> int:
        """Calculate biome-based expansion cost with state type specializations."""
        if native_biome == cell_biome:
            return 5  # Small bonus for native biome familiarity

        # Base biome expansion costs
        biome_base_costs = {
            1: 80,  # Hot desert - challenging
            2: 70,  # Cold desert - difficult
            3: 40,  # Grassland - moderate
            4: 30,  # Temperate grassland - easy
            5: 50,  # Tropical forest - moderate
            6: 45,  # Temperate forest - moderate
            7: 60,  # Boreal forest - harder
            8: 55,  # Temperate rainforest - moderate-hard
            9: 90,  # Wetland - very difficult
            10: 85,  # Tundra - very difficult
            11: 100,  # Glacial - extreme
            12: 0,  # Water - handled separately
        }

        base_cost = biome_base_costs.get(cell_biome, 50)

        # State type biome specializations
        if state_type == "Highland":
            # Highland states adapt well to harsh terrain
            if cell_biome in [1, 2, 10, 11]:  # Desert, tundra, glacial
                base_cost *= 0.7
            elif cell_biome in [7]:  # Boreal forest (mountain-adjacent)
                base_cost *= 0.8

        elif state_type == "Naval":
            # Naval states struggle with inland biomes
            if cell_biome in [1, 2, 10, 11]:  # Harsh inland biomes
                base_cost *= 1.4
            elif cell_biome in [5, 8, 9]:  # Coastal-adjacent biomes
                base_cost *= 0.8

        elif state_type == "River":
            # River states prefer water-adjacent biomes
            if cell_biome in [3, 4, 6]:  # Grassland, temperate forest
                base_cost *= 0.8
            elif cell_biome in [9]:  # Wetlands - river peoples adapt
                base_cost *= 0.6
            elif cell_biome in [1, 2]:  # Deserts - lack rivers
                base_cost *= 1.5

        elif state_type == "Nomadic":
            # Nomadic states excel in grasslands, struggle in forests
            if cell_biome in [3, 4]:  # Grasslands - natural nomad territory
                base_cost *= 0.5
            elif cell_biome in [1, 2]:  # Deserts - nomads can traverse
                base_cost *= 0.7
            elif cell_biome in [5, 6, 7, 8]:  # All forests - major obstacle
                base_cost *= 2.0
            elif cell_biome in [9, 10, 11]:  # Wet/cold - very difficult
                base_cost *= 1.8

        elif state_type == "Lake":
            # Lake cultures prefer temperate, well-watered regions
            if cell_biome in [6, 8, 9]:  # Temperate/wet biomes
                base_cost *= 0.7
            elif cell_biome in [1, 2, 10, 11]:  # Dry/cold biomes
                base_cost *= 1.3

        return int(base_cost)

    def _get_height_cost(self, height: int, state_type: str, cell_id: int) -> int:
        """Calculate height-based expansion cost with state type preferences."""
        # Water crossing penalties (height < 20)
        if height < 20:
            if state_type == "Naval":
                return 150  # Very low penalty for naval states (can project sea power)
            elif state_type == "River":
                return 800  # High penalty - river cultures avoid deep water
            elif state_type == "Lake":
                # Check if it's actually a lake (preferred) vs ocean
                if hasattr(self.features, "feature_ids") and cell_id < len(
                    self.features.feature_ids
                ):
                    feature_id = self.features.feature_ids[cell_id]
                    if feature_id > 0 and feature_id < len(self.features.features):
                        feature = self.features.features[feature_id]
                        if hasattr(feature, "type") and feature.type == "lake":
                            return 5  # Very low penalty for lake expansion
                return 600  # Moderate penalty for ocean expansion
            elif state_type == "Highland":
                return 2000  # Very high penalty - highland peoples avoid water
            elif state_type == "Nomadic":
                return self.options.nomadic_sea_penalty  # Extreme penalty
            else:
                return self.options.sea_crossing_penalty

        # Elevation preferences for different state types
        if state_type == "Highland":
            # Highland states have inverted preferences
            if height >= 80:
                return 0  # Perfect mountain terrain
            elif height >= 62:
                return 50  # Good highland terrain
            elif height >= 40:
                return 200  # Acceptable hills
            else:
                return 800  # Strong penalty for lowlands

        elif state_type == "Naval":
            # Naval states prefer lower elevations (closer to sea level)
            if height >= 80:
                return 1500  # Very high penalty for mountains
            elif height >= 62:
                return 800  # High penalty for highlands
            elif height >= 40:
                return 200  # Moderate penalty for hills
            else:
                return 0  # No penalty for lowlands

        elif state_type == "River":
            # River states prefer moderate elevations (good for rivers)
            if height >= 80:
                return 1200  # High penalty for mountains (no rivers flow there)
            elif height >= 62:
                return 600  # Moderate penalty for highlands
            elif height >= 40:
                return 100  # Small penalty for hills
            elif height >= 30:
                return 0  # Perfect elevation for rivers
            else:
                return 300  # Penalty for very low areas (swamps/coasts)

        elif state_type == "Nomadic":
            # Nomadic states prefer open terrain (avoid steep areas)
            if height >= 80:
                return 1800  # Very high penalty for mountains
            elif height >= 62:
                return 900  # High penalty for highlands
            elif height >= 40:
                return 300  # Moderate penalty for hills
            else:
                return 0  # Good for open plains

        elif state_type == "Lake":
            # Lake states prefer moderate elevations with water access
            if height >= 80:
                return 1000  # High penalty for mountains
            elif height >= 62:
                return 400  # Moderate penalty for highlands
            elif height >= 40:
                return 100  # Small penalty for hills
            else:
                return 50  # Good for lake shores

        else:  # Generic and other state types
            # Standard height penalties
            if height >= 80:
                return 1200  # High mountain penalty
            elif height >= 62:
                return 600  # Highland penalty
            elif height >= 44:
                return 200  # Hill penalty
            else:
                return 0  # No penalty for lowlands

        return 0

    def _get_river_cost(self, cell_id: int, state_type: str) -> int:
        """Calculate river crossing cost."""
        if state_type == "River":
            return 0  # No penalty for river cultures

        # Get flux to determine river size
        if self.graph.flux is not None and cell_id < len(self.graph.flux):
            flux = self.graph.flux[cell_id]
            return min(max(int(flux / 10), 20), 100)

        return 50  # Default river penalty

    def _get_terrain_cost(self, terrain_type: int, state_type: str) -> int:
        """Calculate terrain type cost."""
        if terrain_type == 1:  # Coastline
            if state_type in ["Naval", "Lake"]:
                return 0
            elif state_type == "Nomadic":
                return 60
            else:
                return 20
        elif terrain_type == 2:  # Near coast
            if state_type in ["Naval", "Nomadic"]:
                return 30
            else:
                return 0
        elif terrain_type != -1:  # Inland
            if state_type in ["Naval", "Lake"]:
                return 100
            else:
                return 0

        return 0

    def normalize_states(self) -> None:
        """Clean up state boundaries to create more natural, culture-respecting shapes."""
        logger.info("Normalizing state boundaries with cultural considerations")

        # First pass: Culture-based boundary adjustment
        self._adjust_cultural_boundaries()

        # Second pass: Geographic boundary smoothing
        self._smooth_geographic_boundaries()

        # Third pass: Standard boundary normalization
        self._standard_boundary_normalization()

    def _adjust_cultural_boundaries(self) -> None:
        """Adjust state boundaries to better respect cultural divisions."""
        for i in range(len(self.cell_state)):
            # Skip water, settlements, and locked states
            if (
                self.graph.heights[i] < 20
                or self.cell_settlement[i] > 0
                or self.states.get(self.cell_state[i], State(0, "", 0, 0)).locked
            ):
                continue

            current_state_id = self.cell_state[i]
            if current_state_id == 0:
                continue

            current_state = self.states[current_state_id]
            current_culture = (
                self.cultures.cell_cultures[i]
                if hasattr(self.cultures, "cell_cultures")
                else 0
            )

            neighbors = (
                self.graph.cell_neighbors[i]
                if i < len(self.graph.cell_neighbors)
                else []
            )
            land_neighbors = [
                n
                for n in neighbors
                if n < len(self.graph.heights) and self.graph.heights[n] >= 20
            ]

            if len(land_neighbors) < 2:
                continue

            # Find neighboring states and their cultural alignment
            cultural_alignment = {}  # state_id -> cultural_match_score
            neighbor_states = {}  # state_id -> count

            for n in land_neighbors:
                n_state_id = self.cell_state[n]
                if n_state_id == 0 or n_state_id == current_state_id:
                    continue

                n_state = self.states.get(n_state_id)
                if not n_state or n_state.locked:
                    continue

                neighbor_states[n_state_id] = neighbor_states.get(n_state_id, 0) + 1

                # Calculate cultural alignment score
                n_culture = (
                    self.cultures.cell_cultures[n]
                    if hasattr(self.cultures, "cell_cultures")
                    else 0
                )

                if n_culture == current_culture and n_culture != 0:  # Same culture
                    cultural_alignment[n_state_id] = (
                        cultural_alignment.get(n_state_id, 0) + 3
                    )
                elif (
                    n_state.culture_id == current_culture and current_culture != 0
                ):  # Same state culture
                    cultural_alignment[n_state_id] = (
                        cultural_alignment.get(n_state_id, 0) + 2
                    )
                elif (
                    current_state.culture_id == n_culture and n_culture != 0
                ):  # Cell culture matches current state
                    cultural_alignment[n_state_id] = (
                        cultural_alignment.get(n_state_id, 0) + 1
                    )
                else:  # No cultural alignment
                    cultural_alignment[n_state_id] = (
                        cultural_alignment.get(n_state_id, 0) + 0
                    )

            # Check if we should switch to a culturally better-aligned neighbor
            if neighbor_states:
                best_state = max(
                    neighbor_states.keys(),
                    key=lambda s: (cultural_alignment.get(s, 0), neighbor_states[s]),
                )
                best_alignment = cultural_alignment.get(best_state, 0)
                current_alignment = (
                    1 if current_state.culture_id == current_culture else 0
                )

                # Switch if significantly better cultural alignment
                if (
                    best_alignment > current_alignment + 1
                    and neighbor_states[best_state] >= 2
                ):
                    self.cell_state[i] = best_state

    def _smooth_geographic_boundaries(self) -> None:
        """Smooth boundaries based on geographic features."""
        for i in range(len(self.cell_state)):
            # Skip water, settlements, and locked states
            if (
                self.graph.heights[i] < 20
                or self.cell_settlement[i] > 0
                or self.states.get(self.cell_state[i], State(0, "", 0, 0)).locked
            ):
                continue

            neighbors = (
                self.graph.cell_neighbors[i]
                if i < len(self.graph.cell_neighbors)
                else []
            )

            # Use geographic features as natural boundaries
            current_height = self.graph.heights[i]
            current_state = self.cell_state[i]

            # Check for natural boundary features
            has_river = (
                self.graph.river_ids is not None
                and i < len(self.graph.river_ids)
                and self.graph.river_ids[i] > 0
            )

            # Rivers and height differences can form natural boundaries
            for n in neighbors:
                if n >= len(self.graph.heights) or self.graph.heights[n] < 20:
                    continue

                n_state = self.cell_state[n]
                if n_state == current_state or n_state == 0:
                    continue

                height_diff = abs(current_height - self.graph.heights[n])

                # Major height differences should maintain boundaries
                if height_diff > 30:  # Significant elevation change
                    continue  # Keep separate states

                # Rivers can form natural boundaries but also unite riverine peoples
                if has_river:
                    n_has_river = (
                        self.graph.river_ids is not None
                        and n < len(self.graph.river_ids)
                        and self.graph.river_ids[n] > 0
                    )

                    if n_has_river:
                        # Both have rivers - river states should be unified
                        n_state_obj = self.states.get(n_state)
                        current_state_obj = self.states.get(current_state)

                        if (
                            n_state_obj
                            and current_state_obj
                            and n_state_obj.type == "River"
                            and current_state_obj.type == "River"
                        ):
                            # Consider unifying river states
                            pass  # Could implement river state unification logic

    def _standard_boundary_normalization(self) -> None:
        """Apply standard boundary normalization (original algorithm)."""
        for i in range(len(self.cell_state)):
            # Skip water, settlements, and locked states
            if (
                self.graph.heights[i] < 20
                or self.cell_settlement[i] > 0
                or self.states.get(self.cell_state[i], State(0, "", 0, 0)).locked
            ):
                continue

            # Skip cells near capitals
            neighbors = (
                self.graph.cell_neighbors[i]
                if i < len(self.graph.cell_neighbors)
                else []
            )
            near_capital = False
            for n in neighbors:
                if n < len(self.cell_settlement):
                    settlement_id = self.cell_settlement[n]
                    if settlement_id > 0:
                        settlement = self.settlements.get(settlement_id)
                        if settlement and settlement.is_capital:
                            near_capital = True
                            break

            if near_capital:
                continue

            # Count neighboring states
            current_state = self.cell_state[i]
            land_neighbors = [
                n
                for n in neighbors
                if n < len(self.graph.heights) and self.graph.heights[n] >= 20
            ]

            if len(land_neighbors) < 2:
                continue

            # Count adversaries and buddies
            adversaries = []
            buddies = []

            for n in land_neighbors:
                n_state = self.cell_state[n]
                n_state_obj = self.states.get(n_state)

                if n_state_obj and not n_state_obj.locked:
                    if n_state != current_state:
                        adversaries.append(n)
                    else:
                        buddies.append(n)

            # Switch state if surrounded by adversaries
            if (
                len(adversaries) >= 2
                and len(buddies) <= 2
                and len(adversaries) > len(buddies)
            ):
                self.cell_state[i] = self.cell_state[adversaries[0]]

    def specify_settlements(self) -> None:
        """Calculate settlement properties and features."""
        logger.info("Specifying settlement details")

        for settlement in self.settlements.values():
            if settlement.id == 0:  # Skip placeholder
                continue

            cell_id = settlement.cell_id

            # Determine port status
            settlement.port_id = self._determine_port_status(settlement)

            # Calculate population
            base_pop = max(
                self.cell_suitability[cell_id] / 8
                + settlement.id / 1000
                + (cell_id % 100) / 1000,
                0.1,
            )

            if settlement.is_capital:
                base_pop *= self.options.capital_pop_multiplier

            if settlement.port_id > 0:
                base_pop *= self.options.port_pop_multiplier
                # Adjust position for port
                settlement.x, settlement.y = self._get_port_position(settlement)

            # Add random variation
            gauss_factor = np.clip(np.random.normal(2, 3), 0.6, 20) / 3
            settlement.population = round(base_pop * gauss_factor, 3)

            # Determine settlement type
            settlement.type = self._get_settlement_type(settlement)

            # Assign features based on population
            self._assign_settlement_features(settlement)

            # Generate name using culture-appropriate burg name
            settlement.name = self.name_generator.generate_burg_name(
                settlement.culture_id
            )

    def _determine_port_status(self, settlement: Settlement) -> int:
        """Determine if settlement is a port and which water body."""
        cell_id = settlement.cell_id

        # Check if cell has harbor access
        if not self.graph.cell_haven is not None or cell_id >= len(
            self.graph.cell_haven
        ):
            return 0

        haven = self.graph.cell_haven[cell_id]
        if haven <= 0:
            return 0

        # Check temperature (no frozen ports)
        if self.graph.temperature is not None and self.graph.temperature[cell_id] <= 0:
            return 0

        # Check water body
        if haven < len(self.features.features):
            feature = self.features.features[haven]
            if hasattr(feature, "cells") and feature.cells > 1:
                # Capital with any harbor OR town with good harbor
                if settlement.is_capital and self.graph.harbor_scores is not None:
                    if self.graph.harbor_scores[cell_id] > 0:
                        return feature.id
                elif (
                    self.graph.harbor_scores is not None
                    and self.graph.harbor_scores[cell_id] == 1
                ):
                    return feature.id

        return 0

    def _get_port_position(self, settlement: Settlement) -> Tuple[float, float]:
        """
        Calculate port position near water edge.

        Finds the closest water edge and positions the port there for realistic harbor placement.
        """
        cell_id = settlement.cell_id
        current_x, current_y = settlement.x, settlement.y

        # If no distance field available, return current position
        if not self.graph.distance_field is not None:
            return current_x, current_y

        # Find the best water edge direction
        best_water_pos = None
        min_distance_to_water = float("inf")

        # Check neighbors for water edges
        if self.graph.neighbors is not None and cell_id < len(self.graph.neighbors):
            for neighbor_id in self.graph.neighbors[cell_id]:
                if neighbor_id >= len(self.graph.heights):
                    continue

                # Check if neighbor is water (height < 20)
                if self.graph.heights[neighbor_id] < 20:
                    neighbor_x, neighbor_y = self.graph.points[neighbor_id]
                    distance = (
                        (current_x - neighbor_x) ** 2 + (current_y - neighbor_y) ** 2
                    ) ** 0.5

                    if distance < min_distance_to_water:
                        min_distance_to_water = distance
                        # Position port 2/3 of the way toward the water
                        port_x = current_x + 0.67 * (neighbor_x - current_x)
                        port_y = current_y + 0.67 * (neighbor_y - current_y)
                        best_water_pos = (port_x, port_y)

        # If we found a good water position, use it
        if best_water_pos is not None:
            return best_water_pos

        # Fallback: Use distance field to find water direction
        try:
            distance_field = self.graph.distance_field
            if (
                cell_id < len(distance_field) and distance_field[cell_id] == 1
            ):  # LAND_COAST
                # Look for nearby water cells systematically
                search_radius = min(10, len(self.graph.points) // 100)
                for radius in range(1, search_radius + 1):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx * dx + dy * dy > radius * radius:
                                continue

                            # Estimate cell position (simplified grid approach)
                            if self.graph.cells_x is not None:
                                cells_x = self.graph.cells_x
                                row = cell_id // cells_x
                                col = cell_id % cells_x

                                new_row = row + dy
                                new_col = col + dx

                                if (
                                    0 <= new_row < (len(self.graph.points) // cells_x)
                                    and 0 <= new_col < cells_x
                                ):
                                    test_id = new_row * cells_x + new_col

                                    if (
                                        test_id < len(self.graph.heights)
                                        and self.graph.heights[test_id] < 20
                                    ):
                                        # Found water - move port toward it
                                        test_x, test_y = self.graph.points[test_id]
                                        port_x = current_x + 0.5 * (test_x - current_x)
                                        port_y = current_y + 0.5 * (test_y - current_y)
                                        return (port_x, port_y)

        except (AttributeError, IndexError):
            pass

        # Final fallback: slight offset toward bottom-right (conventional harbor position)
        return (current_x + 1.0, current_y + 1.0)

    def _get_settlement_type(self, settlement: Settlement) -> str:
        """
        Determine settlement type based on location and culture.

        Combines geographic factors with cultural preferences to determine
        appropriate settlement architecture and features.
        """
        cell_id = settlement.cell_id
        culture_id = settlement.culture_id

        # Get culture type for cultural modifiers
        culture_type = "Generic"
        if culture_id in self.cultures.cultures:
            culture_type = self.cultures.cultures[culture_id].type

        # Port settlements (highest priority)
        if settlement.port_id > 0:
            return "Port"  # More specific than just "Naval"

        # Lake settlements
        if self.graph.cell_haven is not None and cell_id < len(self.graph.cell_haven):
            haven = self.graph.cell_haven[cell_id]
            if haven > 0 and haven < len(self.features.features):
                feature = self.features.features[haven]
                if hasattr(feature, "type") and feature.type == "lake":
                    return "Lake"

        # Highland settlements (culture-specific variations)
        if self.graph.heights[cell_id] > 60:
            if culture_type == "Highland":
                return (
                    "Highland Fortress"  # Highland cultures build fortified settlements
                )
            else:
                return "Highland"

        # River settlements (culture-specific variations)
        if (
            self.graph.river_ids is not None
            and self.graph.river_ids[cell_id] > 0
            and self.graph.flux is not None
            and self.graph.flux[cell_id] >= 100
        ):
            if culture_type == "River":
                return "River Trade Hub"  # River cultures specialize in trade
            else:
                return "River"

        # Naval culture coastal settlements
        if culture_type == "Naval":
            # Check if near coast
            if (
                self.graph.distance_field is not None
                and cell_id < len(self.graph.distance_field)
                and self.graph.distance_field[cell_id] <= 2
            ):  # Close to coast
                return "Coastal"

        # Biome and culture-based types
        if hasattr(self.biomes, "cell_biomes") and cell_id < len(
            self.biomes.cell_biomes
        ):
            biome = self.biomes.cell_biomes[cell_id]
            pop = self.cell_population[cell_id]

            # Desert settlements
            if biome in [1, 2]:  # Hot/Cold desert
                if culture_type == "Nomadic":
                    return "Oasis Camp" if pop < 20 else "Desert Town"
                else:
                    return "Desert Outpost"

            # Forest settlements
            elif biome in [5, 6, 7, 8]:  # Forest biomes
                if culture_type == "Hunting":
                    return "Forest Lodge" if pop < 15 else "Woodland Settlement"
                else:
                    return "Forest"

            # Grassland settlements
            elif biome == 4:  # Grassland
                if culture_type == "Nomadic":
                    return "Nomadic Camp" if pop < 10 else "Trading Post"
                else:
                    return "Farming Village"

            # Tundra settlements
            elif biome in [10, 11]:  # Tundra/Glacier
                return "Northern Outpost" if pop < 20 else "Frontier Town"

        # Culture-specific default types
        culture_defaults = {
            "Naval": "Coastal Village",
            "Highland": "Mountain Village",
            "Lake": "Lakeside Village",
            "River": "Riverside Village",
            "Nomadic": "Settlement",
            "Hunting": "Hunting Lodge",
        }

        return culture_defaults.get(culture_type, "Village")

    def _assign_settlement_features(self, settlement: Settlement) -> None:
        """
        Assign settlement features based on population, culture, and settlement type.

        Uses culture-specific architectural preferences and geographic factors.
        """
        pop = settlement.population
        culture_id = settlement.culture_id
        settlement_type = settlement.type

        # Get culture type for modifiers
        culture_type = "Generic"
        if culture_id in self.cultures.cultures:
            culture_type = self.cultures.cultures[culture_id].type

        # Culture-specific feature modifiers
        citadel_bonus = 0.0
        plaza_bonus = 0.0
        walls_bonus = 0.0

        # Highland cultures favor fortifications
        if culture_type == "Highland":
            citadel_bonus = 0.3
            walls_bonus = 0.4
        # Naval cultures favor trade plazas
        elif culture_type == "Naval":
            plaza_bonus = 0.2
            walls_bonus = -0.1  # Less walls, more open trade
        # River cultures favor trade infrastructure
        elif culture_type == "River":
            plaza_bonus = 0.25
        # Nomadic cultures have fewer permanent structures
        elif culture_type == "Nomadic":
            citadel_bonus = -0.2
            walls_bonus = -0.3
            plaza_bonus = -0.1
        # Hunting cultures have modest settlements
        elif culture_type == "Hunting":
            citadel_bonus = -0.1
            walls_bonus = -0.2

        # Settlement type modifiers
        if "Fortress" in settlement_type:
            citadel_bonus += 0.5
            walls_bonus += 0.3
        elif "Trade" in settlement_type or "Hub" in settlement_type:
            plaza_bonus += 0.3
        elif "Camp" in settlement_type or "Lodge" in settlement_type:
            citadel_bonus -= 0.2
            walls_bonus -= 0.2
        elif "Port" in settlement_type:
            plaza_bonus += 0.2
            walls_bonus -= 0.1

        # Citadel - fortified keep/castle
        base_citadel_chance = (
            1.0
            if settlement.is_capital
            else 0.75 if pop > 50 else 0.5 if pop > 15 else 0.1
        )
        citadel_chance = min(1.0, max(0.0, base_citadel_chance + citadel_bonus))
        settlement.citadel = np.random.random() < citadel_chance

        # Plaza - central market/gathering area
        base_plaza_chance = (
            1.0 if pop > 20 else 0.8 if pop > 10 else 0.7 if pop > 4 else 0.6
        )
        plaza_chance = min(1.0, max(0.0, base_plaza_chance + plaza_bonus))
        settlement.plaza = np.random.random() < plaza_chance

        # Walls - defensive fortifications
        base_walls_chance = (
            1.0
            if settlement.is_capital
            else 1.0 if pop > 30 else 0.75 if pop > 20 else 0.5 if pop > 10 else 0.1
        )
        walls_chance = min(1.0, max(0.0, base_walls_chance + walls_bonus))
        settlement.walls = np.random.random() < walls_chance

        # Shantytown - overcrowded poor districts
        # Less likely in nomadic/hunting cultures
        shanty_modifier = -0.3 if culture_type in ["Nomadic", "Hunting"] else 0.0
        base_shanty_chance = (
            1.0
            if pop > 60
            else 0.75 if pop > 40 else 0.4 if (pop > 20 and settlement.walls) else 0.0
        )
        shanty_chance = min(1.0, max(0.0, base_shanty_chance + shanty_modifier))
        settlement.shanty = np.random.random() < shanty_chance

        # Temple assignment - enhanced with religion consideration
        settlement_religion = self._get_religion_for_cell(settlement.cell_id)

        # Base temple probability based on population
        base_temple_chance = (
            0.95
            if pop > 50
            else 0.75 if pop > 35 else 0.5 if pop > 20 else 0.25 if pop > 10 else 0.1
        )

        # Religion influence on temple probability
        religion_modifier = 0.0
        if settlement_religion > 0:
            # Settlements with religions are more likely to have temples
            religion_modifier = 0.2

            # Capital cities are very likely to have major temples
            if settlement.is_capital:
                religion_modifier = 0.4

            # Larger settlements with religions almost always have temples
            if pop > 30:
                religion_modifier = 0.3
        else:
            # Settlements without dominant religion less likely to have formal temples
            religion_modifier = -0.15

        # Culture type modifiers
        if settlement.culture_id in self.cultures.cultures:
            culture_type = self.cultures.cultures[settlement.culture_id].type
            if culture_type == "Highland":
                religion_modifier += 0.1  # Highland cultures value religious centers
            elif culture_type == "Nomadic":
                religion_modifier -= (
                    0.1  # Nomadic cultures have fewer permanent temples
                )

        final_temple_chance = min(
            0.95, max(0.05, base_temple_chance + religion_modifier)
        )
        settlement.temple = np.random.random() < final_temple_chance

    def _get_religion_for_cell(self, cell_id: int) -> int:
        """
        Get the dominant religion for a cell.

        Queries the cell_religions array to determine which religion
        is dominant in the given cell location.

        Args:
            cell_id: Cell index

        Returns:
            Religion ID (0 if no religion or cell out of bounds)
        """
        if cell_id < 0 or cell_id >= len(self.cell_religions):
            return 0

        return int(self.cell_religions[cell_id])
