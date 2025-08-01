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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq

import numpy as np
from sklearn.neighbors import KDTree
import structlog

logger = structlog.get_logger()


@dataclass
class SettlementOptions:
    """Settlement generation options matching FMG's parameters."""
    states_number: int = 30  # Target number of states
    manors_number: int = 1000  # Target number of towns (1000 = auto)
    growth_rate: float = 1.0  # Global growth rate multiplier
    states_growth_rate: float = 1.0  # States-specific growth multiplier
    size_variety: float = 3.0  # Variation in state expansionism
    min_state_cells: int = 10  # Minimum cells for valid state
    
    # Spacing parameters
    capital_spacing_divisor: int = 2  # Divide map size by state count * this
    town_spacing_base: int = 150  # Base divisor for town spacing
    town_spacing_power: float = 0.7  # Power adjustment for town count
    
    # Cost parameters
    culture_same_bonus: int = -9  # Bonus for expanding into same culture
    culture_foreign_penalty: int = 100  # Penalty for expanding into foreign culture
    sea_crossing_penalty: int = 1000  # General sea crossing penalty
    nomadic_sea_penalty: int = 10000  # Extreme penalty for nomadic sea crossing
    
    # Population parameters
    urbanization_rate: float = 0.1  # Target ~10% urbanization
    capital_pop_multiplier: float = 1.3  # Capital population boost
    port_pop_multiplier: float = 1.3  # Port population boost


@dataclass
class Settlement:
    """Data structure for a settlement (burg)."""
    id: int
    cell_id: int
    x: float
    y: float
    name: str = ""
    population: float = 0.0
    is_capital: bool = False
    state_id: int = 0
    culture_id: int = 0
    port_id: int = 0  # Feature ID if port, 0 otherwise
    type: str = "Generic"
    
    # Features
    citadel: bool = False
    plaza: bool = False
    walls: bool = False
    shanty: bool = False
    temple: bool = False


@dataclass  
class State:
    """Data structure for a political state."""
    id: int
    name: str
    capital_id: int
    culture_id: int
    expansionism: float = 1.0
    color: str = "#000000"
    type: str = "Generic"  # Cultural type affecting expansion
    center_cell: int = 0
    territory_cells: List[int] = field(default_factory=list)
    removed: bool = False
    locked: bool = False


class Settlements:
    """Handles settlement placement and state generation."""
    
    def __init__(
        self,
        graph,
        features,
        cultures,
        biomes,
        options: Optional[SettlementOptions] = None,
    ) -> None:
        """
        Initialize Settlements with graph, features, cultures, and biomes data.
        
        Args:
            graph: VoronoiGraph instance with populated data
            features: Features instance with detected geographic features
            cultures: Cultures instance with cultural regions
            biomes: Biomes instance with biome classifications
            options: SettlementOptions for configuration
        """
        self.graph = graph
        self.features = features
        self.cultures = cultures
        self.biomes = biomes
        self.options = options or SettlementOptions()
        
        # Initialize arrays
        self.cell_suitability = np.zeros(len(graph.points), dtype=np.int16)
        self.cell_population = np.zeros(len(graph.points), dtype=np.float32)
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
        
        # Step 1: Calculate cell suitability scores
        self.rank_cells()
        
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
        
        logger.info(f"Generated {len(self.settlements)} settlements and {len(self.states)} states")
        return self.settlements, self.states
    
    def rank_cells(self) -> None:
        """
        Calculate cell suitability scores for settlement placement.
        
        Matches FMG's rankCells function exactly.
        """
        logger.info("Ranking cells for settlement suitability")
        
        # Get flux statistics for normalization
        flux_values = self.graph.flux if hasattr(self.graph, 'flux') else np.zeros(len(self.graph.points))
        confluence_values = self.graph.confluences if hasattr(self.graph, 'confluences') else np.zeros(len(self.graph.points))
        
        # Calculate mean and max flux for normalization
        land_flux = flux_values[self.graph.heights >= 20]
        fl_mean = np.median(land_flux[land_flux > 0]) if len(land_flux) > 0 else 0
        fl_max = np.max(flux_values) + np.max(confluence_values) if len(flux_values) > 0 else 1
        
        # Area normalization
        area_mean = np.mean(self.graph.cell_areas) if hasattr(self.graph, 'cell_areas') else 1.0
        
        for i in range(len(self.graph.points)):
            # Skip water cells
            if self.graph.heights[i] < 20:
                continue
                
            # Get biome habitability as base suitability
            biome_id = self.biomes.cell_biomes[i] if hasattr(self.biomes, 'cell_biomes') else 1
            habitability = self.biomes.get_habitability(biome_id) if hasattr(self.biomes, 'get_habitability') else 100
            
            if habitability == 0:
                continue  # Uninhabitable biomes
                
            s = float(habitability)
            
            # Rivers and confluences are highly valued
            if fl_mean > 0 and i < len(flux_values):
                flux_score = self._normalize(flux_values[i] + confluence_values[i], fl_mean, fl_max) * 250
                s += flux_score
            
            # Low elevation is valued, high is not
            s -= (self.graph.heights[i] - 50) / 5
            
            # Coastal and lake shores get bonuses
            if hasattr(self.graph, 'cell_types') and i < len(self.graph.cell_types):
                if self.graph.cell_types[i] == 1:  # Coastline
                    if hasattr(self.graph, 'river_ids') and self.graph.river_ids[i] > 0:
                        s += 15  # Estuary bonus
                    
                    # Check if it's a lake shore
                    if hasattr(self.graph, 'cell_haven') and i < len(self.graph.cell_haven):
                        haven = self.graph.cell_haven[i]
                        if haven > 0 and haven < len(self.features.features):
                            feature = self.features.features[haven]
                            if hasattr(feature, 'type') and feature.type == "lake":
                                if hasattr(feature, 'freshwater') and feature.freshwater:
                                    s += 30  # Freshwater lake bonus
                                else:
                                    s += 10  # Salt lake bonus
                            elif hasattr(feature, 'type') and feature.type == "ocean":
                                s += 25  # Ocean access bonus
                else:
                    s -= 5  # Non-coastal penalty
            
            # Store suitability score (clamped to int16 range)
            self.cell_suitability[i] = max(0, min(int(s), 32767))
            
            # Calculate population based on suitability and area
            if hasattr(self.graph, 'cell_areas') and i < len(self.graph.cell_areas):
                area_factor = self.graph.cell_areas[i] / area_mean
            else:
                area_factor = 1.0
                
            self.cell_population[i] = max(0, s * area_factor / 100)
    
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
            if (scores[i] > 0 and 
                self.graph.heights[i] >= 20 and
                hasattr(self.cultures, 'cell_cultures') and
                self.cultures.cell_cultures[i] > 0):
                valid_cells.append((scores[i], i))
        
        # Sort by score (highest first)
        valid_cells.sort(reverse=True, key=lambda x: x[0])
        
        if len(valid_cells) < count * 10:
            # Adjust count if not enough valid cells
            count = max(1, len(valid_cells) // 10)
            logger.warning(f"Not enough populated cells. Reducing states to {count}")
        
        # Calculate initial spacing
        map_size = (self.graph.width + self.graph.height) / 2
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
                    culture_id=self.cultures.cell_cultures[cell_id] if hasattr(self.cultures, 'cell_cultures') else 0
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
                logger.warning(f"Retrying capital placement with reduced spacing: {spacing:.2f}")
        
        logger.info(f"Placed {len(capitals)} capital cities")
        return capitals
    
    def create_states(self, capitals: List[Settlement]) -> None:
        """Create initial states from placed capitals."""
        logger.info("Creating states from capitals")
        
        # Create neutral state
        self.states[0] = State(
            id=0,
            name="Neutrals",
            capital_id=0,
            culture_id=0,
            color="#808080"
        )
        
        for capital in capitals:
            state_id = self.next_state_id
            
            # Assign expansionism factor
            expansionism = np.random.random() * self.options.size_variety + 1.0
            
            # Create state
            state = State(
                id=state_id,
                name=f"State {state_id}",  # TODO: Generate proper names
                capital_id=capital.id,
                culture_id=capital.culture_id,
                expansionism=round(expansionism, 1),
                center_cell=capital.cell_id,
                type=self._get_culture_type(capital.culture_id)
            )
            
            self.states[state_id] = state
            capital.state_id = state_id
            self.cell_state[capital.cell_id] = state_id
            self.next_state_id += 1
    
    def _get_culture_type(self, culture_id: int) -> str:
        """Get culture type that affects state expansion."""
        # TODO: Implement culture type determination
        # For now, return generic type
        return "Generic"
    
    def place_towns(self) -> None:
        """Place secondary settlements based on suitability scores."""
        logger.info("Placing towns")
        
        # Calculate scores with more variation for towns
        gauss_factors = np.clip(np.random.normal(1, 3, len(self.cell_suitability)), 0, 20) / 3
        scores = self.cell_suitability * gauss_factors
        
        # Filter for valid town locations (not already occupied)
        valid_cells = []
        for i in range(len(scores)):
            if (self.cell_settlement[i] == 0 and
                scores[i] > 0 and
                self.graph.heights[i] >= 20 and
                hasattr(self.cultures, 'cell_cultures') and
                self.cultures.cell_cultures[i] > 0):
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
        
        # Calculate spacing
        spacing = (self.graph.width + self.graph.height) / self.options.town_spacing_base
        spacing /= (towns_number ** self.options.town_spacing_power / 66)
        
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
                    culture_id=self.cultures.cell_cultures[cell_id] if hasattr(self.cultures, 'cell_cultures') else 0,
                    state_id=0  # Will be assigned during expansion
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
        growth_rate = (len(self.graph.points) / 2 * 
                      self.options.growth_rate * 
                      self.options.states_growth_rate)
        
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
            if hasattr(self.cultures, 'cultures') and state.culture_id in self.cultures.cultures:
                culture = self.cultures.cultures[state.culture_id]
                if hasattr(culture, 'center'):
                    center_cell = culture.center
                    if hasattr(self.biomes, 'cell_biomes') and center_cell < len(self.biomes.cell_biomes):
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
            neighbors = self.graph.cell_neighbors[current_cell] if current_cell < len(self.graph.cell_neighbors) else []
            
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
                    neighbor, state_id, state.culture_id, 
                    state.type, native_biome
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
        native_biome: int
    ) -> float:
        """Calculate cost for state to expand into a cell."""
        cost = 0
        
        # Culture cost
        cell_culture = self.cultures.cell_cultures[cell_id] if hasattr(self.cultures, 'cell_cultures') else 0
        if cell_culture == state_culture:
            cost += self.options.culture_same_bonus
        else:
            cost += self.options.culture_foreign_penalty
        
        # Population cost
        if self.graph.heights[cell_id] < 20:
            cost += 0  # No population penalty for water
        elif self.cell_suitability[cell_id] > 0:
            cost += max(20 - self.cell_suitability[cell_id], 0)
        else:
            cost += 5000  # High penalty for unsuitable land
        
        # Biome cost
        cell_biome = self.biomes.cell_biomes[cell_id] if hasattr(self.biomes, 'cell_biomes') else 1
        cost += self._get_biome_cost(native_biome, cell_biome, state_type)
        
        # Height/terrain cost
        height = self.graph.heights[cell_id]
        cost += self._get_height_cost(height, state_type, cell_id)
        
        # River cost
        if hasattr(self.graph, 'river_ids') and self.graph.river_ids[cell_id] > 0:
            cost += self._get_river_cost(cell_id, state_type)
        
        # Terrain type cost
        if hasattr(self.graph, 'cell_types'):
            terrain_type = self.graph.cell_types[cell_id]
            cost += self._get_terrain_cost(terrain_type, state_type)
        
        return max(cost, 0)
    
    def _get_biome_cost(self, native_biome: int, cell_biome: int, state_type: str) -> int:
        """Calculate biome-based expansion cost."""
        if native_biome == cell_biome:
            return 10  # Small penalty for native biome
        
        # Get base biome cost
        base_cost = 50  # Default cost
        if hasattr(self.biomes, 'get_expansion_cost'):
            base_cost = self.biomes.get_expansion_cost(cell_biome)
        
        # Adjust for state type
        if state_type == "Hunting":
            return base_cost * 2
        elif state_type == "Nomadic" and 4 < cell_biome < 10:  # Forest penalty for nomads
            return base_cost * 3
        
        return base_cost
    
    def _get_height_cost(self, height: int, state_type: str, cell_id: int) -> int:
        """Calculate height-based expansion cost."""
        # Water crossing penalties
        if height < 20:
            if state_type == "Naval":
                return 300  # Low penalty for naval states
            elif state_type == "Nomadic":
                return self.options.nomadic_sea_penalty
            elif state_type == "Lake":
                # Check if it's actually a lake
                if hasattr(self.features, 'feature_ids') and cell_id < len(self.features.feature_ids):
                    feature_id = self.features.feature_ids[cell_id]
                    if feature_id > 0 and feature_id < len(self.features.features):
                        feature = self.features.features[feature_id]
                        if hasattr(feature, 'type') and feature.type == "lake":
                            return 10  # Low lake crossing penalty
                return self.options.sea_crossing_penalty
            else:
                return self.options.sea_crossing_penalty
        
        # Highland preferences
        if state_type == "Highland":
            if height < 62:
                return 1100  # Penalty for lowlands
            else:
                return 0  # No penalty for highlands
        
        # General terrain penalties
        if height >= 67:
            return 2200  # Mountain penalty
        elif height >= 44:
            return 300  # Hill penalty
        
        return 0
    
    def _get_river_cost(self, cell_id: int, state_type: str) -> int:
        """Calculate river crossing cost."""
        if state_type == "River":
            return 0  # No penalty for river cultures
        
        # Get flux to determine river size
        if hasattr(self.graph, 'flux') and cell_id < len(self.graph.flux):
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
        """Clean up state boundaries to create more natural shapes."""
        logger.info("Normalizing state boundaries")
        
        for i in range(len(self.cell_state)):
            # Skip water, settlements, and locked states
            if (self.graph.heights[i] < 20 or 
                self.cell_settlement[i] > 0 or
                self.states.get(self.cell_state[i], State(0, "", 0, 0)).locked):
                continue
            
            # Skip cells near capitals
            neighbors = self.graph.cell_neighbors[i] if i < len(self.graph.cell_neighbors) else []
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
            land_neighbors = [n for n in neighbors if n < len(self.graph.heights) and self.graph.heights[n] >= 20]
            
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
            if (len(adversaries) >= 2 and 
                len(buddies) <= 2 and 
                len(adversaries) > len(buddies)):
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
            base_pop = max(self.cell_suitability[cell_id] / 8 + settlement.id / 1000 + (cell_id % 100) / 1000, 0.1)
            
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
            
            # Generate name
            settlement.name = f"Settlement {settlement.id}"  # TODO: Implement name generation
    
    def _determine_port_status(self, settlement: Settlement) -> int:
        """Determine if settlement is a port and which water body."""
        cell_id = settlement.cell_id
        
        # Check if cell has harbor access
        if not hasattr(self.graph, 'cell_haven') or cell_id >= len(self.graph.cell_haven):
            return 0
            
        haven = self.graph.cell_haven[cell_id]
        if haven <= 0:
            return 0
        
        # Check temperature (no frozen ports)
        if hasattr(self.graph, 'temperature') and self.graph.temperature[cell_id] <= 0:
            return 0
        
        # Check water body
        if haven < len(self.features.features):
            feature = self.features.features[haven]
            if hasattr(feature, 'cells') and feature.cells > 1:
                # Capital with any harbor OR town with good harbor
                if settlement.is_capital and hasattr(self.graph, 'harbor_scores'):
                    if self.graph.harbor_scores[cell_id] > 0:
                        return feature.id
                elif hasattr(self.graph, 'harbor_scores') and self.graph.harbor_scores[cell_id] == 1:
                    return feature.id
        
        return 0
    
    def _get_port_position(self, settlement: Settlement) -> Tuple[float, float]:
        """Calculate port position near water edge."""
        # TODO: Implement edge point calculation
        # For now, return slight offset
        return settlement.x, settlement.y
    
    def _get_settlement_type(self, settlement: Settlement) -> str:
        """Determine settlement type based on location."""
        cell_id = settlement.cell_id
        
        if settlement.port_id > 0:
            return "Naval"
        
        # Check for lake settlements
        if hasattr(self.graph, 'cell_haven') and cell_id < len(self.graph.cell_haven):
            haven = self.graph.cell_haven[cell_id]
            if haven > 0 and haven < len(self.features.features):
                feature = self.features.features[haven]
                if hasattr(feature, 'type') and feature.type == "lake":
                    return "Lake"
        
        # Highland settlements
        if self.graph.heights[cell_id] > 60:
            return "Highland"
        
        # River settlements
        if (hasattr(self.graph, 'river_ids') and 
            self.graph.river_ids[cell_id] > 0 and
            hasattr(self.graph, 'flux') and 
            self.graph.flux[cell_id] >= 100):
            return "River"
        
        # Check biome-based types
        if hasattr(self.biomes, 'cell_biomes') and cell_id < len(self.biomes.cell_biomes):
            biome = self.biomes.cell_biomes[cell_id]
            pop = self.cell_population[cell_id]
            
            if not settlement.id or pop <= 5:
                if pop < 5 and biome in [1, 2, 3, 4]:  # Desert/tundra biomes
                    return "Nomadic"
                elif 4 < biome < 10:  # Forest biomes
                    return "Hunting"
        
        return "Generic"
    
    def _assign_settlement_features(self, settlement: Settlement) -> None:
        """Assign settlement features based on population."""
        pop = settlement.population
        
        # Citadel
        settlement.citadel = (
            settlement.is_capital or
            (pop > 50 and np.random.random() < 0.75) or
            (pop > 15 and np.random.random() < 0.5) or
            np.random.random() < 0.1
        )
        
        # Plaza
        settlement.plaza = (
            pop > 20 or
            (pop > 10 and np.random.random() < 0.8) or
            (pop > 4 and np.random.random() < 0.7) or
            np.random.random() < 0.6
        )
        
        # Walls
        settlement.walls = (
            settlement.is_capital or
            pop > 30 or
            (pop > 20 and np.random.random() < 0.75) or
            (pop > 10 and np.random.random() < 0.5) or
            np.random.random() < 0.1
        )
        
        # Shantytown
        settlement.shanty = (
            pop > 60 or
            (pop > 40 and np.random.random() < 0.75) or
            (pop > 20 and settlement.walls and np.random.random() < 0.4)
        )
        
        # Temple
        # TODO: Add religion check when available
        settlement.temple = (
            pop > 50 or
            (pop > 35 and np.random.random() < 0.75) or
            (pop > 20 and np.random.random() < 0.5)
        )