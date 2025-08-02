"""
Name generation system using Markov chains.

This module ports the procedural name generation from FMG to Python,
using a custom Markov chain implementation that matches FMG's approach.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from py_fmg.core.alea_prng import AleaPRNG
from py_fmg.core.markov_name_generator import MarkovChain, MarkovNameGenerator


class EntityType(Enum):
    """Types of entities that can have generated names."""

    CULTURE = "culture"
    STATE = "state"
    BURG = "burg"
    RIVER = "river"
    MOUNTAIN = "mountain"
    LAKE = "lake"
    MAP_TITLE = "map_title"


class NameBase(BaseModel):
    """Configuration for a cultural naming pattern."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the name base")
    i: int = Field(description="Index identifier")
    min: int = Field(description="Minimum name length")
    max: int = Field(description="Maximum name length")
    d: str = Field(description="Allowed duplicate letters")
    m: float = Field(description="Multi-word name rate (deprecated in FMG)")
    b: str = Field(description="Base names (comma-separated)")

    def get_names_list(self) -> List[str]:
        """Extract names as a list."""
        return [name.strip() for name in self.b.split(",") if name.strip()]


class NameGenerator:
    """Main name generator using Markov chains."""

    def __init__(self, prng: Optional[AleaPRNG] = None):
        """Initialize name generator with optional PRNG for deterministic generation."""
        self.prng = prng or AleaPRNG(seed="default")
        self.markov = MarkovNameGenerator(self.prng)
        self.name_bases: Dict[int, NameBase] = {}
        self.chains: Dict[int, Optional[MarkovChain]] = {}
        self._load_default_name_bases()

    def _load_default_name_bases(self) -> None:
        """Load the default name bases from FMG."""
        from py_fmg.core.name_bases import load_default_name_bases

        load_default_name_bases(self)

    def add_name_base(self, name_base: NameBase) -> None:
        """Add a new name base configuration."""
        self.name_bases[name_base.i] = name_base
        self.chains[name_base.i] = None  # Clear cached chain

    def _build_chain(self, base_id: int) -> Optional[MarkovChain]:
        """Build a Markov chain for a name base."""
        if base_id not in self.name_bases:
            return None

        name_base = self.name_bases[base_id]
        names = name_base.get_names_list()

        if not names:
            return None

        return self.markov.build_chain(names)

    def _get_chain(self, base_id: int) -> Optional[MarkovChain]:
        """Get or build the Markov chain for a base."""
        if base_id not in self.chains or self.chains[base_id] is None:
            self.chains[base_id] = self._build_chain(base_id)
        return self.chains[base_id]

    def generate_base_name(
        self,
        base_id: int,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        allow_duplicates: Optional[str] = None,
    ) -> str:
        """Generate a name using a specific name base.

        Args:
            base_id: ID of the name base to use
            min_length: Minimum name length (overrides base default)
            max_length: Maximum name length (overrides base default)
            allow_duplicates: Letters allowed to duplicate (overrides base default)

        Returns:
            Generated name string
        """
        if base_id not in self.name_bases:
            # Fall back to first available base
            if self.name_bases:
                base_id = next(iter(self.name_bases))
            else:
                return "ERROR"

        name_base = self.name_bases[base_id]
        chain = self._get_chain(base_id)

        if not chain:
            return "ERROR"

        # Use base defaults if not specified
        min_len = min_length or name_base.min
        max_len = max_length or name_base.max
        dupl = allow_duplicates if allow_duplicates is not None else name_base.d

        # Generate name
        name = self.markov.generate(chain, min_len, max_len, dupl)

        # If generation failed, pick random from base
        if name == "ERROR" and name_base.b:
            names = name_base.get_names_list()
            if names:
                return self.prng.choice(names)

        return name

    def generate_culture_name(
        self, culture_id: int, base_id: Optional[int] = None
    ) -> str:
        """Generate a name for a culture."""
        # In FMG, cultures have associated bases
        # For now, just use the base_id directly
        return self.generate_base_name(base_id or culture_id)

    def generate_culture_short(
        self, culture_id: int, base_id: Optional[int] = None
    ) -> str:
        """Generate a short name for a culture."""
        base = base_id or culture_id
        if base not in self.name_bases:
            return self.generate_base_name(base)

        name_base = self.name_bases[base]
        min_len = max(1, name_base.min - 1)
        max_len = max(min_len, name_base.max - 2)

        return self.generate_base_name(base, min_len, max_len)

    def generate_state_name(
        self, base_name: str, culture_id: int, base_id: Optional[int] = None
    ) -> str:
        """Generate a state name based on capital or culture.

        Applies culture-specific rules for state naming.
        """
        if not base_name:
            return "ERROR"

        name = base_name
        base = base_id if base_id is not None else culture_id

        # Remove spaces (no multi-word state names)
        if " " in name:
            name = name.replace(" ", "").capitalize()

        # Remove certain endings
        if len(name) > 6 and name.endswith("berg"):
            name = name[:-4]
        if len(name) > 5 and name.endswith("ton"):
            name = name[:-3]

        # Apply culture-specific rules based on base ID
        if base == 5:  # Ruthenian
            if name.endswith(("sk", "ev", "ov")):
                name = name[:-2]
        elif (
            base == 12
        ):  # Japanese (not in current bases, but keeping for compatibility)
            if not name[-1].lower() in "aeiou":
                name += "u"
        elif base == 18:  # Arabic (not in current bases)
            if self.prng.random() < 0.4:
                if name[0].lower() in "aeiou":
                    name = "Al" + name.lower()
                else:
                    name = "Al " + name

        # Add culture-specific suffixes
        suffix_map = {
            0: ["burg", "stadt", "dorf", "heim"],  # German
            1: ["ton", "ham", "shire", "bury"],  # English
            2: ["ville", "ac", "y", "agne"],  # French
            3: ["no", "na", "ne", "o"],  # Italian
            4: ["ez", "os", "a", "o"],  # Castillian
            6: ["holm", "stad", "vik", "berg"],  # Nordic
            7: ["polis", "ion", "os", "a"],  # Greek
            8: ["um", "ia", "is", "us"],  # Roman
        }

        if base in suffix_map and self.prng.random() < 0.3:
            suffix = self.prng.choice(suffix_map[base])
            name = self._add_suffix(name, suffix)

        return name

    def _add_suffix(self, name: str, suffix: str) -> str:
        """Add a suffix to a name, handling vowel conflicts."""
        if not name or not suffix:
            return name + suffix

        # If both end and start with vowels, remove final vowel
        if name[-1].lower() in "aeiou" and suffix[0].lower() in "aeiou":
            name = name[:-1]

        return name + suffix

    def generate_burg_name(self, culture_id: int, base_id: Optional[int] = None) -> str:
        """Generate a settlement (burg) name."""
        base = base_id if base_id is not None else culture_id
        base_name = self.generate_base_name(base)

        # Add burg-specific suffixes based on culture
        burg_suffixes = {
            0: ["dorf", "hausen", "heim", "bach", "burg"],  # German
            1: ["ton", "ham", "wick", "thorpe", "by"],  # English
            2: ["ville", "sur", "les", "bains"],  # French
            3: ["monte", "borgo", "castello"],  # Italian
            4: ["nuevo", "viejo", "alto", "bajo"],  # Castillian
            6: ["by", "torp", "vik", "stad"],  # Nordic
            7: ["polis", "chora"],  # Greek
            8: ["castrum", "vicus"],  # Roman
        }

        if base in burg_suffixes and self.prng.random() < 0.4:
            suffix = self.prng.choice(burg_suffixes[base])
            base_name = self._add_suffix(base_name, suffix)

        return base_name

    def generate_river_name(
        self, culture_id: int, base_id: Optional[int] = None
    ) -> str:
        """Generate a river name."""
        base = base_id if base_id is not None else culture_id
        base_name = self.generate_base_name(base, min_length=4, max_length=8)

        # Add river-specific patterns based on culture
        river_patterns = {
            0: ["fluss", "bach", "wasser"],  # German
            1: ["river", "water", "brook"],  # English
            2: ["fleuve", "riviere"],  # French
            3: ["fiume", "torrente"],  # Italian
            4: ["rio"],  # Castillian
            6: ["elv", "aa"],  # Nordic
            7: ["potamos"],  # Greek
            8: ["flumen", "aqua"],  # Roman
        }

        if base in river_patterns and self.prng.random() < 0.3:
            pattern = self.prng.choice(river_patterns[base])
            # Some cultures use prefix, others suffix
            if base in [7, 8]:  # Greek, Roman prefer prefix
                base_name = f"{pattern} {base_name}"
            else:
                base_name = self._add_suffix(base_name, pattern)

        return base_name

    def generate_mountain_name(
        self, culture_id: int, base_id: Optional[int] = None
    ) -> str:
        """Generate a mountain name."""
        base = base_id if base_id is not None else culture_id
        base_name = self.generate_base_name(base, min_length=4, max_length=10)

        # Add mountain-specific patterns
        mountain_patterns = {
            0: ["berg", "spitze", "horn"],  # German
            1: ["mount", "peak", "fell"],  # English
            2: ["mont", "pic", "aiguille"],  # French
            3: ["monte", "cima", "punta"],  # Italian
            4: ["monte", "pico", "sierra"],  # Castillian
            6: ["fjell", "tind"],  # Nordic
            7: ["oros"],  # Greek
            8: ["mons"],  # Roman
        }

        if base in mountain_patterns:
            pattern = self.prng.choice(mountain_patterns[base])
            # Most cultures use prefix for mountains
            if base in [0, 6]:  # German, Nordic sometimes use suffix
                if self.prng.random() < 0.5:
                    base_name = self._add_suffix(base_name, pattern)
                else:
                    base_name = f"{pattern.capitalize()} {base_name}"
            else:
                base_name = f"{pattern.capitalize()} {base_name}"

        return base_name

    def generate_map_title(self) -> str:
        """Generate a title for the map."""
        # Use a random base for variety
        base_id = (
            self.prng.choice(list(self.name_bases.keys())) if self.name_bases else 0
        )

        # Generate shorter name for map title
        if base_id in self.name_bases:
            name_base = self.name_bases[base_id]
            min_len = max(3, name_base.min - 2)
            max_len = max(min_len + 1, min(8, name_base.max - 3))
        else:
            min_len, max_len = 3, 8

        base_name = self.generate_base_name(base_id, min_len, max_len)

        # Add suffix
        if self.prng.random() < 0.8:
            suffix = "ia"
            # Shorten name if needed
            if len(base_name) > 6:
                base_name = base_name[:3]
        else:
            suffix = "land"
            # Shorten name if needed
            if len(base_name) > 6:
                base_name = base_name[:5]

        return self._validate_suffix(base_name, suffix)

    def _validate_suffix(self, name: str, suffix: str) -> str:
        """Validate and add suffix to name (FMG compatibility)."""
        # This matches FMG's validateSuffix function
        if not name:
            return suffix

        # Handle vowel conflicts
        last_char = name[-1].lower()
        first_suffix_char = suffix[0].lower()

        if last_char in "aeiou" and first_suffix_char in "aeiou":
            # Remove last vowel
            name = name[:-1]

        return name + suffix
