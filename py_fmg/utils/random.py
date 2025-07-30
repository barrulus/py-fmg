"""
Random number generation utilities.

This module provides access to FMG's Alea PRNG for exact compatibility
with the JavaScript implementation. Python's random and NumPy's random
should not be used in FMG code.
"""

from ..core.alea_prng import AleaPRNG

# Global PRNG instance
_prng = None


def set_random_seed(seed: str) -> None:
    """
    Set the random seed for FMG's Alea PRNG.
    
    This only sets the seed for the FMG-compatible Alea PRNG.
    Python's random and NumPy's random are not used to ensure
    exact compatibility with the JavaScript FMG implementation.
    
    Args:
        seed: Seed string to use
    """
    global _prng
    
    # Initialize Alea PRNG only
    _prng = AleaPRNG(seed)


def get_prng() -> AleaPRNG:
    """
    Get the current Alea PRNG instance.
    
    Returns:
        AleaPRNG instance
    """
    global _prng
    if _prng is None:
        _prng = AleaPRNG("default")
    return _prng