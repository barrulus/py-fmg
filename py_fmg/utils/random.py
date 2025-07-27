"""
Random number generation utilities.
"""

import random
import numpy as np
from ..core.alea_prng import AleaPRNG

# Global PRNG instance
_prng = None


def set_random_seed(seed: str) -> None:
    """
    Set the random seed for all random number generators.
    
    Args:
        seed: Seed string to use
    """
    global _prng
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(hash(seed) % (2**32))
    
    # Initialize Alea PRNG
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