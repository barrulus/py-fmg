"""
Religion generation system based on FMG's religions.js.

This module implements the full FMG religion system with:
- Folk religions (one per culture)
- Organized religions with expansion mechanics
- Proper temple and sacred site assignment
- Religion-culture-state interactions
"""

from __future__ import annotations
import colorsys
import heapq
import math
import numpy as np
import structlog
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict
from .alea_prng import AleaPRNG
from .name_generator import NameGenerator


logger = structlog.get_logger()


class ReligionOptions(BaseModel):
    """Religion generation options matching FMG's parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    religions_number: int = Field(default=8, description="Target organized religions")
    theocracy_chance: float = Field(
        default=0.1, description="Chance state becomes theocracy"
    )
    temple_pop_threshold: int = Field(
        default=50, description="Guaranteed temple population"
    )
    sacred_site_density: float = Field(
        default=1.0, description="Sacred site generation multiplier"
    )


class Religion(BaseModel):
    """Data structure for a religion matching FMG specification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(description="Unique religion identifier")
    name: str = Field(description="Religion name")
    color: str = Field(description="Religion color in hex format")
    type: str = Field(description="Religion type: Folk, Organized, Cult, Heresy")
    form: str = Field(description="Specific religious form")
    culture_id: int = Field(description="Associated culture ID")
    center: int = Field(description="Cell ID of religion center")
    deity: Optional[str] = Field(default=None, description="Supreme deity name")
    expansion: str = Field(
        default="global", description="Expansion type: global, state, culture"
    )
    expansionism: float = Field(
        default=1.0, description="Expansion competitiveness (0-10)"
    )
    origins: List[int] = Field(default_factory=list, description="Parent religion IDs")
    code: str = Field(default="", description="Abbreviated code")

    # Statistics (calculated)
    cells: Set[int] = Field(
        default_factory=set, description="Cells following this religion"
    )
    area: float = Field(default=0.0, description="Total area covered")
    rural_population: float = Field(
        default=0.0, description="Rural population following"
    )
    urban_population: float = Field(
        default=0.0, description="Urban population following"
    )


class ReligionGenerator:
    """Generates religions following FMG's algorithm."""

    # Religion forms by type (from FMG)
    FOLK_FORMS = [
        "Shamanism",
        "Animism",
        "Polytheism",
        "Ancestor Worship",
        "Nature Worship",
        "Totemism",
    ]
    ORGANIZED_FORMS = ["Polytheism", "Monotheism", "Dualism", "Pantheism", "Non-theism"]
    CULT_FORMS = ["Cult", "Dark Cult", "Sect"]

    # Theocracy state names
    THEOCRACY_FORMS = [
        "Diocese",
        "Bishopric",
        "Eparchy",
        "Exarchate",
        "Patriarchate",
        "Imamah",
        "Caliphate",
    ]

    def __init__(
        self,
        graph,
        cultures_dict: Dict,
        cell_cultures: np.ndarray,
        settlements_dict: Dict,
        states_dict: Dict,
        options: Optional[ReligionOptions] = None,
        prng: Optional[AleaPRNG] = None,
        name_generator: Optional[NameGenerator] = None,
    ):
        """
        Initialize religion generator.

        Args:
            graph: VoronoiGraph with populated data
            cultures_dict: Culture data by ID
            cell_cultures: Culture assignment per cell
            settlements_dict: Settlement data by ID
            states_dict: State data by ID
            options: Religion generation options
            prng: Random number generator
            name_generator: Name generator for culture-based deity names
        """
        self.graph = graph
        self.cultures_dict = cultures_dict
        self.cell_cultures = cell_cultures
        self.settlements_dict = settlements_dict
        self.states_dict = states_dict
        self.options = options or ReligionOptions()
        self.prng = prng or AleaPRNG("religions")
        self.name_generator = name_generator or NameGenerator(prng)

        self.religions: Dict[int, Religion] = {}
        self.cell_religions = np.zeros(len(graph.points), dtype=np.int32)
        self.next_religion_id = 1

        # Color palettes
        self.religion_colors = [
            "#ff0000",
            "#00ff00",
            "#0000ff",
            "#ffff00",
            "#ff00ff",
            "#00ffff",
            "#800000",
            "#008000",
            "#000080",
            "#808000",
            "#800080",
            "#008080",
            "#ffa500",
            "#ffc0cb",
            "#a0522d",
            "#8b4513",
            "#dda0dd",
            "#98fb98",
        ]

    def generate(self) -> Tuple[Dict[int, Religion], np.ndarray]:
        """
        Generate complete religion system.

        Returns:
            Tuple of (religions_dict, cell_religions_array)
        """
        logger.info("Starting religion generation")

        # Step 1: Generate folk religions (one per culture)
        self._generate_folk_religions()

        # Step 2: Generate organized religions
        self._generate_organized_religions()

        # Step 3: Expand organized religions
        self._expand_organized_religions()

        # Step 4: Update states with theocracies
        self._assign_theocracies()

        # Step 5: Calculate statistics
        self._calculate_statistics()

        logger.info(f"Generated {len(self.religions)} religions")
        return self.religions, self.cell_religions

    def _generate_folk_religions(self) -> None:
        """Generate folk religions - one per culture."""
        logger.info("Generating folk religions")

        for culture_id, culture in self.cultures_dict.items():
            religion = Religion(
                id=self.next_religion_id,
                name=self._generate_folk_religion_name(culture),
                color=culture.color,  # Folk religions use culture color
                type="Folk",
                form=self.prng.choice(self.FOLK_FORMS),
                culture_id=culture_id,
                center=culture.center,
                expansion="culture",  # Folk religions limited to culture
                expansionism=0.0,  # Folk religions don't expand
            )

            # Generate deity for theistic forms
            if religion.form not in ["Animism", "Non-theism"]:
                religion.deity = self._generate_deity_name(culture)

            religion.code = self._generate_religion_code(religion)

            self.religions[religion.id] = religion

            # Assign all culture cells to this folk religion initially
            culture_cells = np.where(self.cell_cultures == culture_id)[0]
            for cell_id in culture_cells:
                if cell_id < len(self.cell_religions):
                    self.cell_religions[cell_id] = religion.id
                    religion.cells.add(cell_id)

            self.next_religion_id += 1

    def _generate_organized_religions(self) -> None:
        """Generate organized religions with strategic placement."""
        logger.info("Generating organized religions")

        # Find good centers for organized religions
        potential_centers = self._find_religion_centers()

        # Generate organized religions
        religions_to_generate = min(
            self.options.religions_number, len(potential_centers)
        )

        for i in range(religions_to_generate):
            if i >= len(potential_centers):
                break

            center_cell = potential_centers[i]
            parent_culture_id = int(self.cell_cultures[center_cell])
            parent_culture = self.cultures_dict.get(parent_culture_id)

            if not parent_culture:
                continue

            # Determine religion type and expansionism
            if self.prng.random() < 0.1:
                religion_type = "Cult"
                forms = self.CULT_FORMS
                expansionism = max(0.0, self.prng.gauss(1.0, 0.5))
            else:
                religion_type = "Organized"
                forms = self.ORGANIZED_FORMS
                expansionism = max(0.0, self.prng.gauss(5.0, 3.0))

            expansionism = min(10.0, expansionism)

            religion = Religion(
                id=self.next_religion_id,
                name="",  # Will be generated after form is set
                color=self._generate_religion_color(parent_culture, religion_type),
                type=religion_type,
                form=self.prng.choice(forms),
                culture_id=parent_culture_id,
                center=center_cell,
                expansionism=expansionism,
            )

            # Set expansion scope
            rand = self.prng.random()
            if rand < 0.3:
                religion.expansion = "culture"
            elif rand < 0.6:
                religion.expansion = "state"
            else:
                religion.expansion = "global"

            # Generate deity for theistic forms
            if religion.form not in ["Animism", "Non-theism"]:
                religion.deity = self._generate_deity_name(parent_culture)

            religion.name = self._generate_organized_religion_name(
                religion, parent_culture
            )
            religion.code = self._generate_religion_code(religion)

            self.religions[religion.id] = religion
            self.next_religion_id += 1

    def _find_religion_centers(self) -> List[int]:
        """Find good centers for organized religions based on population and spacing."""
        candidates = []

        # Find populated cells that could be religion centers
        for cell_id in range(len(self.graph.points)):
            if (
                cell_id < len(self.graph.heights) and self.graph.heights[cell_id] >= 20
            ):  # Land only

                # Get population from settlements or culture suitability
                population = 0.0
                if hasattr(self.graph, "cell_population"):
                    population = self.graph.cell_population[cell_id]
                elif hasattr(self.graph, "cell_suitability"):
                    population = self.graph.cell_suitability[cell_id]

                if population > 5:  # Minimum population threshold
                    candidates.append((population, cell_id))

        # Sort by population (descending)
        candidates.sort(reverse=True, key=lambda x: x[0])

        # Apply spacing constraints
        selected_centers = []
        min_spacing = max(10, int(math.sqrt(len(self.graph.points)) / 4))

        for pop, cell_id in candidates:
            # Check spacing from existing centers
            too_close = False
            for existing_center in selected_centers:
                if self._calculate_distance(cell_id, existing_center) < min_spacing:
                    too_close = True
                    break

            if not too_close:
                selected_centers.append(cell_id)

            if len(selected_centers) >= self.options.religions_number * 2:
                break

        return selected_centers[: self.options.religions_number]

    def _expand_organized_religions(self) -> None:
        """Expand organized religions using cost-based algorithm."""
        logger.info("Expanding organized religions")

        organized_religions = [
            r for r in self.religions.values() if r.type in ["Organized", "Cult"]
        ]

        if not organized_religions:
            return

        # Priority queue for expansion (cost, cell_id, religion_id)
        expansion_queue = []

        # Initialize with religion centers
        for religion in organized_religions:
            if religion.expansionism > 0:
                # Replace folk religion at center
                if religion.center < len(self.cell_religions):
                    old_religion_id = self.cell_religions[religion.center]
                    if old_religion_id > 0 and old_religion_id in self.religions:
                        self.religions[old_religion_id].cells.discard(religion.center)

                    self.cell_religions[religion.center] = religion.id
                    religion.cells.add(religion.center)

                    # Add neighbors to queue
                    self._add_neighbors_to_expansion_queue(
                        expansion_queue, religion.center, religion.id, 0.0
                    )

        # Expand religions based on costs
        processed_cells = set()

        while expansion_queue:
            cost, cell_id, religion_id = heapq.heappop(expansion_queue)

            if cell_id in processed_cells:
                continue

            processed_cells.add(cell_id)
            religion = self.religions[religion_id]

            # Check expansion constraints
            if not self._can_expand_to_cell(religion, cell_id):
                continue

            # Calculate final cost with religion's expansionism
            final_cost = cost / max(0.1, religion.expansionism)

            # Competition with existing religion
            current_religion_id = self.cell_religions[cell_id]
            if current_religion_id > 0:
                current_religion = self.religions[current_religion_id]
                current_cost = self._calculate_expansion_cost(
                    current_religion, current_religion.center, cell_id
                ) / max(0.1, current_religion.expansionism)

                # Only expand if we're more competitive
                if final_cost >= current_cost:
                    continue

                # Remove cell from old religion
                current_religion.cells.discard(cell_id)

            # Assign cell to new religion
            self.cell_religions[cell_id] = religion_id
            religion.cells.add(cell_id)

            # Add neighbors for further expansion
            self._add_neighbors_to_expansion_queue(
                expansion_queue, cell_id, religion_id, cost
            )

    def _add_neighbors_to_expansion_queue(
        self, queue: List, cell_id: int, religion_id: int, base_cost: float
    ) -> None:
        """Add neighboring cells to expansion queue."""
        if not hasattr(self.graph, "neighbors") or cell_id >= len(self.graph.neighbors):
            return

        religion = self.religions[religion_id]

        for neighbor_id in self.graph.neighbors[cell_id]:
            if (
                neighbor_id < len(self.graph.heights)
                and self.graph.heights[neighbor_id] >= 20
            ):  # Land only

                expansion_cost = self._calculate_expansion_cost(
                    religion, cell_id, neighbor_id
                )
                total_cost = base_cost + expansion_cost

                heapq.heappush(queue, (total_cost, neighbor_id, religion_id))

    def _calculate_expansion_cost(
        self, religion: Religion, from_cell: int, to_cell: int
    ) -> float:
        """Calculate cost for religion to expand to a cell."""
        base_cost = 1.0

        # Culture cost
        from_culture = (
            self.cell_cultures[from_cell] if from_cell < len(self.cell_cultures) else 0
        )
        to_culture = (
            self.cell_cultures[to_cell] if to_cell < len(self.cell_cultures) else 0
        )

        if from_culture != to_culture:
            base_cost += 10.0  # Different culture penalty
        else:
            base_cost -= 9.0  # Same culture bonus

        # State boundary cost (simplified - would need state cell mapping)
        # For now, use culture as proxy for state boundaries
        if religion.expansion != "state" and from_culture != to_culture:
            base_cost += 10.0

        # Terrain cost (water is impassable for land religions)
        if self.graph.heights[to_cell] < 20:
            base_cost += 1000.0

        return max(0.1, base_cost)

    def _can_expand_to_cell(self, religion: Religion, cell_id: int) -> bool:
        """Check if religion can expand to given cell."""
        # Water check
        if self.graph.heights[cell_id] < 20:
            return False

        # Expansion scope check
        if religion.expansion == "culture":
            cell_culture = (
                self.cell_cultures[cell_id] if cell_id < len(self.cell_cultures) else 0
            )
            return cell_culture == religion.culture_id

        # For "state" and "global", allow expansion (state checking would need state-cell mapping)
        return True

    def _assign_theocracies(self) -> None:
        """Assign theocracy government forms to states based on religions."""
        logger.info("Assigning theocracies")

        for state in self.states_dict.values():
            if hasattr(state, "removed") and state.removed:
                continue

            # Find dominant religion in state
            dominant_religion = self._find_dominant_religion_in_state(state)

            if dominant_religion and dominant_religion.type in ["Organized", "Cult"]:
                # Check for theocracy
                theocracy_chance = self.options.theocracy_chance
                if dominant_religion.expansion == "state":
                    theocracy_chance = 1.0  # Guaranteed for state religions

                if self.prng.random() < theocracy_chance:
                    state.type = self.prng.choice(self.THEOCRACY_FORMS)

    def _find_dominant_religion_in_state(self, state) -> Optional[Religion]:
        """Find the dominant religion in a state."""
        if not hasattr(state, "territory_cells") or not state.territory_cells:
            return None

        religion_counts = {}
        for cell_id in state.territory_cells:
            if cell_id < len(self.cell_religions):
                religion_id = self.cell_religions[cell_id]
                if religion_id > 0:
                    religion_counts[religion_id] = (
                        religion_counts.get(religion_id, 0) + 1
                    )

        if not religion_counts:
            return None

        dominant_id = max(religion_counts.keys(), key=lambda x: religion_counts[x])
        return self.religions.get(dominant_id)

    def _calculate_statistics(self) -> None:
        """Calculate area and population statistics for religions."""
        logger.info("Calculating religion statistics")

        for religion in self.religions.values():
            religion.area = len(religion.cells)
            religion.rural_population = 0.0
            religion.urban_population = 0.0

            for cell_id in religion.cells:
                if hasattr(self.graph, "cell_population") and cell_id < len(
                    self.graph.cell_population
                ):
                    population = self.graph.cell_population[cell_id]

                    # Check if cell has settlement
                    has_settlement = any(
                        s.cell_id == cell_id for s in self.settlements_dict.values()
                    )

                    if has_settlement:
                        religion.urban_population += population
                    else:
                        religion.rural_population += population

    def _calculate_distance(self, cell1: int, cell2: int) -> float:
        """Calculate distance between two cells."""
        if cell1 >= len(self.graph.points) or cell2 >= len(self.graph.points):
            return float("inf")

        x1, y1 = self.graph.points[cell1]
        x2, y2 = self.graph.points[cell2]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _generate_folk_religion_name(self, culture) -> str:
        """Generate name for folk religion."""
        forms = {
            "Shamanism": f"{culture.name} Shamanism",
            "Animism": f"{culture.name} Spirits",
            "Polytheism": f"{culture.name} Pantheon",
            "Ancestor Worship": f"{culture.name} Ancestors",
            "Nature Worship": f"{culture.name} Nature Faith",
            "Totemism": f"{culture.name} Totems",
        }
        return forms.get("Generic", f"{culture.name} Beliefs")

    def _generate_organized_religion_name(self, religion: Religion, culture) -> str:
        """Generate name for organized religion."""
        if religion.form == "Monotheism" and religion.deity:
            templates = [
                f"Faith of {religion.deity}",
                f"Church of {religion.deity}",
                f"{religion.deity}ism",
            ]
            return self.prng.choice(templates)
        elif religion.form == "Polytheism":
            return f"{culture.name} Pantheon"
        elif religion.form in ["Cult", "Dark Cult"]:
            if religion.deity:
                return f"Cult of {religion.deity}"
            else:
                return f"{culture.name} Cult"
        else:
            return f"{culture.name}ian {religion.form}"

    def _generate_deity_name(self, culture) -> str:
        """Generate deity name using FMG's culture-based system."""
        # Use NameGenerator with culture's name base (like FMG's Names.getCulture)
        # This matches FMG's approach: Names.getCulture(culture, null, null, "", 0.8)
        culture_name = self.name_generator.generate_base_name(
            culture.name_base, min_length=4, max_length=8
        )

        meaning = self._generate_deity_meaning()
        return f"{culture_name}, The {meaning}"

    def _generate_deity_meaning(self) -> str:
        """Generate deity meaning using FMG's approach system."""
        # FMG deity name generation approaches with weights
        approaches = []
        approach_weights = {
            "Number": 1,
            "Being": 3,
            "Adjective": 5,
            "Color + Animal": 5,
            "Adjective + Animal": 5,
            "Adjective + Being": 5,
            "Adjective + Genitive": 1,
            "Color + Being": 3,
            "Color + Genitive": 3,
            "Being + of + Genitive": 2,
            "Being + of the + Genitive": 1,
            "Animal + of + Genitive": 1,
            "Adjective + Being + of + Genitive": 2,
            "Adjective + Animal + of + Genitive": 2,
        }

        # Create weighted list like FMG
        for approach, weight in approach_weights.items():
            approaches.extend([approach] * weight)

        # FMG base terms for meaning generation
        base_terms = {
            "number": [
                "One",
                "Two",
                "Three",
                "Four",
                "Five",
                "Six",
                "Seven",
                "Eight",
                "Nine",
                "Ten",
                "Eleven",
                "Twelve",
            ],
            "being": [
                "Ancestor",
                "Ancient",
                "Avatar",
                "Brother",
                "Champion",
                "Chief",
                "Council",
                "Creator",
                "Deity",
                "Divine One",
                "Elder",
                "Enlightened Being",
                "Father",
                "Forebear",
                "Forefather",
                "Giver",
                "God",
                "Goddess",
                "Guardian",
                "Guide",
                "Hierach",
                "Lady",
                "Lord",
                "Maker",
                "Master",
                "Mother",
                "Numen",
                "Oracle",
                "Overlord",
                "Protector",
                "Reaper",
                "Ruler",
                "Sage",
                "Seer",
                "Sister",
                "Spirit",
                "Supreme Being",
                "Transcendent",
                "Virgin",
            ],
            "animal": [
                "Antelope",
                "Ape",
                "Badger",
                "Basilisk",
                "Bear",
                "Beaver",
                "Bison",
                "Boar",
                "Buffalo",
                "Camel",
                "Cat",
                "Centaur",
                "Cerberus",
                "Chimera",
                "Cobra",
                "Cockatrice",
                "Crane",
                "Crocodile",
                "Crow",
                "Cyclope",
                "Deer",
                "Dog",
                "Direwolf",
                "Drake",
                "Dragon",
                "Eagle",
                "Elephant",
                "Elk",
                "Falcon",
                "Fox",
                "Goat",
                "Goose",
                "Gorgon",
                "Gryphon",
                "Hare",
                "Hawk",
                "Heron",
                "Hippogriff",
                "Horse",
                "Hound",
                "Hyena",
                "Ibis",
                "Jackal",
                "Jaguar",
                "Kitsune",
                "Kraken",
                "Lark",
                "Leopard",
                "Lion",
                "Manticore",
                "Mantis",
                "Marten",
                "Minotaur",
                "Moose",
                "Mule",
                "Narwhal",
                "Owl",
                "Ox",
                "Panther",
                "Pegasus",
                "Phoenix",
                "Python",
                "Rat",
                "Raven",
                "Roc",
                "Rook",
                "Scorpion",
                "Serpent",
                "Shark",
                "Sheep",
                "Snake",
                "Sphinx",
                "Spider",
                "Swan",
                "Tiger",
                "Turtle",
                "Unicorn",
                "Viper",
                "Vulture",
                "Walrus",
                "Wolf",
                "Wolverine",
                "Worm",
                "Wyvern",
                "Yeti",
            ],
            "adjective": [
                "Aggressive",
                "Almighty",
                "Ancient",
                "Beautiful",
                "Benevolent",
                "Big",
                "Blind",
                "Blond",
                "Bloody",
                "Brave",
                "Broken",
                "Brutal",
                "Burning",
                "Calm",
                "Celestial",
                "Cheerful",
                "Crazy",
                "Cruel",
                "Dead",
                "Deadly",
                "Devastating",
                "Distant",
                "Disturbing",
                "Divine",
                "Dying",
                "Eternal",
                "Ethernal",
                "Empyreal",
                "Enigmatic",
                "Enlightened",
                "Evil",
                "Explicit",
                "Fair",
                "Far",
                "Fat",
                "Fatal",
                "Favorable",
                "Flying",
                "Friendly",
                "Frozen",
                "Giant",
                "Good",
                "Grateful",
                "Great",
                "Happy",
                "High",
                "Holy",
                "Honest",
                "Huge",
                "Hungry",
                "Illustrious",
                "Immutable",
                "Ineffable",
                "Infallible",
                "Inherent",
                "Last",
                "Latter",
                "Lost",
                "Loud",
                "Lucky",
                "Mad",
                "Magical",
                "Main",
                "Major",
                "Marine",
                "Mythical",
                "Mystical",
                "Naval",
                "New",
                "Noble",
                "Old",
                "Otherworldly",
                "Patient",
                "Peaceful",
                "Pregnant",
                "Prime",
                "Proud",
                "Pure",
                "Radiant",
                "Resplendent",
                "Sacred",
                "Sacrosanct",
                "Sad",
                "Scary",
                "Secret",
                "Selected",
                "Serene",
                "Severe",
                "Silent",
                "Sleeping",
                "Slumbering",
                "Sovereign",
                "Strong",
                "Sunny",
                "Superior",
                "Supernatural",
                "Sustainable",
                "Transcendent",
                "Transcendental",
                "Troubled",
                "Unearthly",
                "Unfathomable",
                "Unhappy",
                "Unknown",
                "Unseen",
                "Waking",
                "Wild",
                "Wise",
                "Worried",
                "Young",
            ],
            "genitive": [
                "Cold",
                "Day",
                "Death",
                "Doom",
                "Fate",
                "Fire",
                "Fog",
                "Frost",
                "Gates",
                "Heaven",
                "Home",
                "Ice",
                "Justice",
                "Life",
                "Light",
                "Lightning",
                "Love",
                "Nature",
                "Night",
                "Pain",
                "Snow",
                "Springs",
                "Summer",
                "Thunder",
                "Time",
                "Victory",
                "War",
                "Winter",
            ],
            "the_genitive": [
                "Abyss",
                "Blood",
                "Dawn",
                "Earth",
                "East",
                "Eclipse",
                "Fall",
                "Harvest",
                "Moon",
                "North",
                "Peak",
                "Rainbow",
                "Sea",
                "Sky",
                "South",
                "Stars",
                "Storm",
                "Sun",
                "Tree",
                "Underworld",
                "West",
                "Wild",
                "Word",
                "World",
            ],
            "color": [
                "Amber",
                "Black",
                "Blue",
                "Bright",
                "Bronze",
                "Brown",
                "Coral",
                "Crimson",
                "Dark",
                "Emerald",
                "Golden",
                "Green",
                "Grey",
                "Indigo",
                "Lavender",
                "Light",
                "Magenta",
                "Maroon",
                "Orange",
                "Pink",
                "Plum",
                "Purple",
                "Red",
                "Ruby",
                "Sapphire",
                "Teal",
                "Turquoise",
                "White",
                "Yellow",
            ],
        }

        # Select approach and generate meaning like FMG
        approach = self.prng.choice(approaches)

        if approach == "Number":
            return self.prng.choice(base_terms["number"])
        elif approach == "Being":
            return self.prng.choice(base_terms["being"])
        elif approach == "Adjective":
            return self.prng.choice(base_terms["adjective"])
        elif approach == "Color + Animal":
            return f"{self.prng.choice(base_terms['color'])} {self.prng.choice(base_terms['animal'])}"
        elif approach == "Adjective + Animal":
            return f"{self.prng.choice(base_terms['adjective'])} {self.prng.choice(base_terms['animal'])}"
        elif approach == "Adjective + Being":
            return f"{self.prng.choice(base_terms['adjective'])} {self.prng.choice(base_terms['being'])}"
        elif approach == "Adjective + Genitive":
            return f"{self.prng.choice(base_terms['adjective'])} {self.prng.choice(base_terms['genitive'])}"
        elif approach == "Color + Being":
            return f"{self.prng.choice(base_terms['color'])} {self.prng.choice(base_terms['being'])}"
        elif approach == "Color + Genitive":
            return f"{self.prng.choice(base_terms['color'])} {self.prng.choice(base_terms['genitive'])}"
        elif approach == "Being + of + Genitive":
            return f"{self.prng.choice(base_terms['being'])} of {self.prng.choice(base_terms['genitive'])}"
        elif approach == "Being + of the + Genitive":
            return f"{self.prng.choice(base_terms['being'])} of the {self.prng.choice(base_terms['the_genitive'])}"
        elif approach == "Animal + of + Genitive":
            return f"{self.prng.choice(base_terms['animal'])} of {self.prng.choice(base_terms['genitive'])}"
        elif approach == "Adjective + Being + of + Genitive":
            return f"{self.prng.choice(base_terms['adjective'])} {self.prng.choice(base_terms['being'])} of {self.prng.choice(base_terms['genitive'])}"
        elif approach == "Adjective + Animal + of + Genitive":
            return f"{self.prng.choice(base_terms['adjective'])} {self.prng.choice(base_terms['animal'])} of {self.prng.choice(base_terms['genitive'])}"
        else:
            # Fallback
            return self.prng.choice(base_terms["being"])

    def _generate_religion_code(self, religion: Religion) -> str:
        """Generate abbreviated code for religion."""
        words = religion.name.split()
        if len(words) >= 2:
            return (words[0][:2] + words[1][:2]).upper()
        else:
            return religion.name[:4].upper()

    def _generate_religion_color(
        self, parent_culture, religion_type: str = "Organized"
    ) -> str:
        """
        Generate color for religion based on parent culture using color theory.

        Different religion types get different mixing parameters:
        - Folk: Uses pure culture color
        - Organized: Mixed with base colors, moderate brightness
        - Cult: Mixed with more variation, darker
        - Heresy: Mixed with culture color, less brightness variation
        """
        # Folk religions use pure culture color
        if religion_type == "Folk":
            return parent_culture.color

        # Get base color for mixing
        base_color = self.prng.choice(self.religion_colors)

        try:
            # Type-specific mixing parameters based on FMG's approach
            if religion_type == "Heresy":
                # Heresy: close to culture color with subtle variation
                return self._mix_colors_advanced(
                    parent_culture.color,
                    base_color,
                    culture_weight=0.65,  # More culture influence
                    brightness_variation=0.2,  # Less brightness change
                )
            elif religion_type == "Cult":
                # Cult: more dramatic departure, often darker
                return self._mix_colors_advanced(
                    parent_culture.color,
                    base_color,
                    culture_weight=0.5,  # Balanced mix
                    brightness_variation=0.0,  # No brightness boost
                )
            else:  # Organized
                # Organized: moderate mixing with brightness boost
                return self._mix_colors_advanced(
                    parent_culture.color,
                    base_color,
                    culture_weight=0.25,  # Less culture influence
                    brightness_variation=0.4,  # More brightness variation
                )
        except (ValueError, IndexError):
            return base_color

    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB values (0-1 range)."""
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]

        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        return r, g, b

    def _rgb_to_hex(self, r: float, g: float, b: float) -> str:
        """Convert RGB values (0-1 range) to hex color."""
        r = max(0, min(255, int(r * 255)))
        g = max(0, min(255, int(g * 255)))
        b = max(0, min(255, int(b * 255)))

        return f"#{r:02x}{g:02x}{b:02x}"

    def _mix_colors_advanced(
        self,
        color1: str,
        color2: str,
        culture_weight: float = 0.3,
        brightness_variation: float = 0.3,
    ) -> str:
        """
        Mix two colors using HSV color space for better visual harmony.

        Based on FMG's getMixedColor but enhanced with color theory:
        - Uses HSV space for hue preservation
        - Maintains color harmony while providing variation
        - Avoids muddy RGB mixing results

        Args:
            color1: First color (culture color)
            color2: Second color (base religion color)
            culture_weight: How much of culture color to preserve (0-1)
            brightness_variation: Brightness adjustment (-1 to 1)

        Returns:
            Mixed color as hex string
        """
        # Convert to RGB
        r1, g1, b1 = self._hex_to_rgb(color1)
        r2, g2, b2 = self._hex_to_rgb(color2)

        # Convert to HSV for better color mixing
        h1, s1, v1 = colorsys.rgb_to_hsv(r1, g1, b1)
        h2, s2, v2 = colorsys.rgb_to_hsv(r2, g2, b2)

        # Mix hues with special handling for wraparound
        hue_diff = abs(h1 - h2)
        if hue_diff > 0.5:  # Handle hue wraparound (e.g., red to purple)
            # Choose the shorter path around the color wheel
            if h1 > h2:
                h2 += 1.0
            else:
                h1 += 1.0

        mixed_hue = (h1 * culture_weight + h2 * (1 - culture_weight)) % 1.0

        # Mix saturation and value (brightness)
        mixed_saturation = s1 * culture_weight + s2 * (1 - culture_weight)
        mixed_value = v1 * culture_weight + v2 * (1 - culture_weight)

        # Apply brightness variation similar to FMG's .brighter() function
        if brightness_variation > 0:
            # Brighten: increase value and decrease saturation slightly
            mixed_value = min(1.0, mixed_value * (1 + brightness_variation))
            mixed_saturation = max(
                0.1, mixed_saturation * (1 - brightness_variation * 0.2)
            )
        elif brightness_variation < 0:
            # Darken: decrease value and increase saturation slightly
            mixed_value = max(0.1, mixed_value * (1 + brightness_variation))
            mixed_saturation = min(
                1.0, mixed_saturation * (1 - brightness_variation * 0.2)
            )

        # Add slight random variation for uniqueness (like FMG's random mixing)
        hue_variation = (self.prng.random() - 0.5) * 0.05  # Small hue shift
        mixed_hue = (mixed_hue + hue_variation) % 1.0

        # Convert back to RGB
        mixed_r, mixed_g, mixed_b = colorsys.hsv_to_rgb(
            mixed_hue, mixed_saturation, mixed_value
        )

        return self._rgb_to_hex(mixed_r, mixed_g, mixed_b)

    def assign_temples_to_settlements(self, settlements_dict: Dict) -> None:
        """Assign temples to settlements based on religion and population."""
        logger.info("Assigning temples to settlements")

        for settlement in settlements_dict.values():
            if settlement.cell_id >= len(self.cell_religions):
                continue

            religion_id = self.cell_religions[settlement.cell_id]
            if religion_id == 0:
                continue

            religion = self.religions.get(religion_id)
            if not religion:
                continue

            # Check if settlement's state is a theocracy
            theocracy_bonus = False
            if (
                hasattr(settlement, "state_id")
                and settlement.state_id in self.states_dict
            ):
                state = self.states_dict[settlement.state_id]
                theocracy_bonus = (
                    hasattr(state, "type") and state.type in self.THEOCRACY_FORMS
                )

            # Temple assignment logic from FMG
            pop = settlement.population

            temple_chance = 0.0
            if pop > self.options.temple_pop_threshold:
                temple_chance = 1.0  # Guaranteed
            elif pop > 35:
                temple_chance = 0.75
            elif pop > 20:
                temple_chance = 0.5

            # Theocracy bonus
            if theocracy_bonus:
                temple_chance = min(1.0, temple_chance + 0.5)

            settlement.temple = self.prng.random() < temple_chance
