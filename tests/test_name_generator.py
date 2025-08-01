"""Tests for the name generation system."""

import pytest

from py_fmg.core.alea_prng import AleaPRNG
from py_fmg.core.name_generator import EntityType, NameBase, NameGenerator


class TestNameGenerator:
    """Test the name generator functionality."""
    
    def test_initialization(self):
        """Test that name generator initializes with default bases."""
        gen = NameGenerator()
        
        # Should have loaded default name bases
        assert len(gen.name_bases) > 0
        assert 0 in gen.name_bases  # German
        assert 1 in gen.name_bases  # English
    
    def test_deterministic_generation(self):
        """Test that using the same seed produces same results."""
        # Create two generators with same seed
        prng1 = AleaPRNG(seed=42)
        gen1 = NameGenerator(prng=prng1)
        
        prng2 = AleaPRNG(seed=42)
        gen2 = NameGenerator(prng=prng2)
        
        # Generate names
        names1 = [gen1.generate_base_name(0) for _ in range(5)]
        names2 = [gen2.generate_base_name(0) for _ in range(5)]
        
        assert names1 == names2
    
    def test_base_name_generation(self):
        """Test basic name generation for each culture."""
        gen = NameGenerator()
        
        # Test first few cultures
        for base_id in range(min(5, len(gen.name_bases))):
            name = gen.generate_base_name(base_id)
            assert name != "ERROR"
            assert len(name) >= 2
            
            # Check it respects min/max lengths
            base = gen.name_bases[base_id]
            assert len(name) >= base.min
            assert len(name) <= base.max
    
    def test_custom_length_constraints(self):
        """Test that custom length constraints are respected."""
        gen = NameGenerator()
        
        # Generate short names
        for _ in range(10):
            name = gen.generate_base_name(0, min_length=3, max_length=5)
            assert 3 <= len(name) <= 5
        
        # Generate long names
        for _ in range(10):
            name = gen.generate_base_name(0, min_length=10, max_length=15)
            assert 10 <= len(name) <= 15
    
    def test_duplicate_handling(self):
        """Test that duplicate letter rules are applied."""
        gen = NameGenerator()
        
        # German allows 'l' and 't' duplicates
        german_base = gen.name_bases[0]
        assert german_base.d == "lt"
        
        # Generate names and check
        for _ in range(20):
            name = gen.generate_base_name(0)
            # Check no triple letters
            for i in range(len(name) - 2):
                assert not (name[i] == name[i+1] == name[i+2])
    
    def test_culture_name_generation(self):
        """Test culture name generation."""
        gen = NameGenerator()
        
        # Generate culture names
        for culture_id in range(3):
            name = gen.generate_culture_name(culture_id)
            assert name != "ERROR"
            assert len(name) >= 2
    
    def test_state_name_generation(self):
        """Test state name generation with cultural rules."""
        gen = NameGenerator()
        
        # Test German state
        state = gen.generate_state_name("Frankfurt", 0, 0)
        assert state != "ERROR"
        assert "Frankfurt" in state or len(state) > 0
        
        # Test that -berg endings are removed
        state = gen.generate_state_name("Heidelberg", 0, 0)
        assert not state.endswith("bergberg")
        
        # Test Ruthenian removes -sk/-ev/-ov
        state = gen.generate_state_name("Minsk", 5, 5)
        assert not state.endswith("sk")
    
    def test_burg_name_generation(self):
        """Test settlement name generation."""
        gen = NameGenerator()
        
        # Generate burg names for different cultures
        for culture_id in range(3):
            name = gen.generate_burg_name(culture_id)
            assert name != "ERROR"
            assert len(name) >= 2
    
    def test_river_name_generation(self):
        """Test river name generation."""
        gen = NameGenerator()
        
        # Generate river names
        for culture_id in range(3):
            name = gen.generate_river_name(culture_id)
            assert name != "ERROR"
            assert len(name) >= 2
    
    def test_mountain_name_generation(self):
        """Test mountain name generation."""
        gen = NameGenerator()
        
        # Generate mountain names
        for culture_id in range(3):
            name = gen.generate_mountain_name(culture_id)
            assert name != "ERROR"
            assert len(name) >= 2
    
    def test_map_title_generation(self):
        """Test map title generation."""
        gen = NameGenerator()
        
        # Generate map titles
        for _ in range(10):
            title = gen.generate_map_title()
            assert title != "ERROR"
            assert len(title) >= 2
            # Should end with 'ia' or 'land'
            assert title.endswith("ia") or title.endswith("land")
    
    def test_custom_name_base(self):
        """Test adding custom name bases."""
        gen = NameGenerator()
        
        # Add custom name base
        custom_base = NameBase(
            name="Custom",
            i=100,
            min=4,
            max=8,
            d="",
            m=0,
            b="Alpha,Beta,Gamma,Delta,Epsilon,Zeta,Eta,Theta"
        )
        
        gen.add_name_base(custom_base)
        
        # Generate names from custom base
        name = gen.generate_base_name(100)
        assert name != "ERROR"
        assert 4 <= len(name) <= 8
    
    def test_fallback_behavior(self):
        """Test fallback when generation fails."""
        gen = NameGenerator()
        
        # Request non-existent base
        name = gen.generate_base_name(999)
        # Should fall back to first available base
        assert name != "ERROR"
    
    def test_empty_generator(self):
        """Test behavior with no name bases."""
        gen = NameGenerator()
        gen.name_bases.clear()
        
        name = gen.generate_base_name(0)
        assert name == "ERROR"