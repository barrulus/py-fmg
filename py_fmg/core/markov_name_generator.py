"""Markov chain-based name generator that matches FMG's implementation.

This module implements the syllable-based Markov chain approach used by FMG
for generating fantasy names.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from py_fmg.core.alea_prng import AleaPRNG


@dataclass
class MarkovChain:
    """Markov chain for name generation."""
    
    data: Dict[str, List[str]]
    
    @classmethod
    def from_names(cls, names: List[str]) -> MarkovChain:
        """Build a Markov chain from a list of names.
        
        This follows FMG's approach of building chains from syllables.
        """
        chain: Dict[str, List[str]] = {}
        
        for name in names:
            if not name:
                continue
                
            # Split into syllables (FMG's approach)
            syllables = cls._split_into_syllables(name.lower())
            
            # Build chain from syllables
            prev = ""  # Start token
            
            for syllable in syllables:
                if prev not in chain:
                    chain[prev] = []
                chain[prev].append(syllable)
                prev = syllable
            
            # Add end token
            if prev not in chain:
                chain[prev] = []
            chain[prev].append("")  # End token
        
        return cls(data=chain)
    
    @staticmethod
    def _split_into_syllables(name: str) -> List[str]:
        """Split a name into syllables following FMG's approach.
        
        This is a simplified version that approximates FMG's syllable detection.
        """
        if not name:
            return []
        
        syllables = []
        current = ""
        vowels = set("aeiou")
        
        for i, char in enumerate(name):
            current += char
            
            # Simple syllable detection rules
            if char in vowels:
                # Check if next char is consonant or end
                if i + 1 >= len(name) or name[i + 1] not in vowels:
                    # Check if we should continue the syllable
                    if i + 2 < len(name) and name[i + 1] in "lrn" and name[i + 2] not in vowels:
                        # Include consonant cluster
                        current += name[i + 1]
                        if i + 2 < len(name):
                            current += name[i + 2]
                            i += 2
                    elif i + 1 < len(name) and name[i + 1] not in vowels:
                        # Include single consonant
                        current += name[i + 1]
                        i += 1
                    
                    syllables.append(current)
                    current = ""
            elif i == len(name) - 1:
                # Last character
                if current:
                    if syllables:
                        # Append to last syllable
                        syllables[-1] += current
                    else:
                        syllables.append(current)
        
        return syllables


class MarkovNameGenerator:
    """Name generator using Markov chains, matching FMG's implementation."""
    
    def __init__(self, prng: Optional[AleaPRNG] = None):
        """Initialize the generator with optional PRNG."""
        self.prng = prng or AleaPRNG(seed="default")
        self.chains: Dict[int, Optional[MarkovChain]] = {}
    
    def build_chain(self, names: List[str]) -> MarkovChain:
        """Build a Markov chain from a list of names."""
        return MarkovChain.from_names(names)
    
    def generate(
        self,
        chain: MarkovChain,
        min_length: int = 5,
        max_length: int = 12,
        duplicates: str = "",
        max_attempts: int = 20,
    ) -> str:
        """Generate a name using the Markov chain.
        
        Args:
            chain: The Markov chain to use
            min_length: Minimum name length
            max_length: Maximum name length
            duplicates: Letters allowed to duplicate
            max_attempts: Maximum generation attempts
            
        Returns:
            Generated name string
        """
        for attempt in range(max_attempts):
            name = self._generate_attempt(chain, min_length, max_length)
            
            if name and min_length <= len(name) <= max_length:
                # Process the name
                processed = self._process_name(name, duplicates)
                if len(processed) >= 2:  # Ensure minimum viable length
                    return processed
        
        # Fallback: return random element from chain
        return self._get_fallback_name(chain)
    
    def _generate_attempt(
        self, chain: MarkovChain, min_length: int, max_length: int
    ) -> str:
        """Generate a single name attempt."""
        if not chain.data or "" not in chain.data:
            return ""
        
        result = ""
        current = ""  # Start token
        
        for _ in range(20):  # Max iterations
            if current not in chain.data:
                break
            
            options = chain.data[current]
            if not options:
                break
            
            # Select next syllable
            next_syllable = self.prng.choice(options)
            
            if next_syllable == "":  # End token
                if len(result) >= min_length:
                    break
                else:
                    # Too short, restart
                    current = ""
                    result = ""
                    continue
            
            # Check if adding would exceed max length
            if len(result) + len(next_syllable) > max_length:
                if len(result) >= min_length:
                    break
                else:
                    # Can't fit, break anyway
                    result += next_syllable
                    break
            
            result += next_syllable
            current = next_syllable
        
        return result
    
    def _process_name(self, name: str, duplicates: str) -> str:
        """Process a generated name according to FMG rules."""
        if not name:
            return ""
        
        # Remove trailing special characters
        while name and name[-1] in ["'", " ", "-"]:
            name = name[:-1]
        
        # Check if this is a multi-word name (has spaces)
        has_spaces = " " in name
        
        # Process character by character
        result = []
        prev_char = ""
        
        for i, char in enumerate(name):
            # First character is uppercase
            if i == 0:
                result.append(char.upper())
                prev_char = char
                continue
            
            # Remove duplicates unless allowed
            # For multi-word names, preserve duplicates within words
            if char == prev_char and char not in duplicates:
                # If we have spaces and neither char is a space, skip duplicate removal
                if not (has_spaces and char != ' ' and prev_char != ' '):
                    continue
            
            # Remove three same letters in a row
            if i >= 2 and char == name[i-1] == name[i-2]:
                continue
            
            # Capitalize after space or hyphen
            if prev_char in [" ", "-"]:
                result.append(char.upper())
                prev_char = char.upper()
            else:
                result.append(char)
                prev_char = char
        
        final_name = "".join(result)
        
        # Join if any part has only 1 letter
        if " " in final_name:
            parts = final_name.split(" ")
            if any(len(part) < 2 for part in parts):
                # Join all parts
                final_name = "".join(p.capitalize() if i == 0 else p.lower() 
                                   for i, p in enumerate(parts))
        
        return final_name
    
    def _get_fallback_name(self, chain: MarkovChain) -> str:
        """Get a fallback name from the chain data."""
        # Find all complete paths in the chain
        all_syllables = set()
        for syllables in chain.data.values():
            all_syllables.update(s for s in syllables if s)
        
        if all_syllables:
            # Create a simple name from syllables
            syllable = self.prng.choice(list(all_syllables))
            return syllable.capitalize()
        
        return "ERROR"