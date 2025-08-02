"""Tests for the Markov chain name generator."""

import pytest

from py_fmg.core.alea_prng import AleaPRNG
from py_fmg.core.markov_name_generator import MarkovChain, MarkovNameGenerator


class TestMarkovChain:
    """Test the Markov chain functionality."""
    
    def test_chain_from_names(self):
        """Test building a chain from names."""
        names = ["test", "tent", "text"]
        chain = MarkovChain.from_names(names)
        
        # Should have data
        assert chain.data
        assert "" in chain.data  # Start token
    
    def test_syllable_splitting(self):
        """Test syllable splitting logic."""
        # Test simple cases
        syllables = MarkovChain._split_into_syllables("hello")
        assert len(syllables) > 0
        
        syllables = MarkovChain._split_into_syllables("computer")
        assert len(syllables) > 0
        
        # Test empty string
        syllables = MarkovChain._split_into_syllables("")
        assert syllables == []


class TestMarkovNameGenerator:
    """Test the Markov name generator."""
    
    def test_generation(self):
        """Test basic name generation."""
        gen = MarkovNameGenerator()
        
        # Build chain from sample names
        names = ["Alice", "Albert", "Alexandra", "Alexander", "Alison"]
        chain = gen.build_chain(names)
        
        # Generate names
        for _ in range(10):
            name = gen.generate(chain, min_length=3, max_length=10)
            assert name != "ERROR"
            assert 3 <= len(name) <= 10
            # Should start with capital letter
            assert name[0].isupper()
    
    def test_deterministic_generation(self):
        """Test deterministic generation with seed."""
        prng1 = AleaPRNG(seed=123)
        gen1 = MarkovNameGenerator(prng=prng1)
        
        prng2 = AleaPRNG(seed=123)
        gen2 = MarkovNameGenerator(prng=prng2)
        
        names = ["John", "Jane", "Jack", "Jill", "James"]
        chain1 = gen1.build_chain(names)
        chain2 = gen2.build_chain(names)
        
        # Generate names
        names1 = [gen1.generate(chain1) for _ in range(5)]
        names2 = [gen2.generate(chain2) for _ in range(5)]
        
        assert names1 == names2
    
    def test_duplicate_removal(self):
        """Test that duplicates are removed correctly."""
        gen = MarkovNameGenerator()
        
        # Test with no allowed duplicates
        processed = gen._process_name("hello", "")
        assert "ll" not in processed.lower()
        
        # Test with allowed duplicates
        processed = gen._process_name("hello", "l")
        assert "ll" in processed.lower()
    
    def test_name_processing(self):
        """Test name processing rules."""
        gen = MarkovNameGenerator()
        
        # Test trailing character removal
        assert gen._process_name("test'", "") == "Test"
        assert gen._process_name("test-", "") == "Test"
        assert gen._process_name("test ", "") == "Test"
        
        # Test capitalization after space
        assert gen._process_name("hello world", "") == "Hello World"
        assert gen._process_name("hello-world", "") == "Hello-World"
    
    def test_edge_cases(self):
        """Test edge cases."""
        gen = MarkovNameGenerator()
        
        # Empty chain
        chain = MarkovChain(data={})
        name = gen.generate(chain)
        assert name == "ERROR"
        
        # Single name chain
        names = ["Test"]
        chain = gen.build_chain(names)
        name = gen.generate(chain)
        assert name != "ERROR"