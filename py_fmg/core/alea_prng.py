"""
Python implementation of Alea PRNG to match FMG's random number generation.

Based on Johannes Baagøe's Alea algorithm used in FMG.
This ensures exact reproducibility of FMG's jittered grids.
"""


def _uint32(n):
    """Convert to unsigned 32-bit integer."""
    return int(n) & 0xFFFFFFFF


class AleaPRNG:
    """
    Alea PRNG implementation matching the JavaScript version used in FMG.

    This is a direct port of the JavaScript Alea algorithm.
    """

    def __init__(self, seed):
        """Initialize with seed string or number."""
        # Add call counter
        self.call_count = 0

        # Convert arguments to array
        if hasattr(seed, "__iter__") and not isinstance(seed, str):
            args = list(seed)
        else:
            args = [seed]

        # Mash function
        mash_n = 0xEFC8249D  # 4022871197

        def mash(data):
            nonlocal mash_n
            data = str(data)
            for char in data:
                mash_n = mash_n + ord(char)
                h = 0.02519603282416938 * mash_n
                mash_n = _uint32(h)
                h -= mash_n
                h *= mash_n
                mash_n = _uint32(h)
                h -= mash_n
                mash_n += h * 0x100000000  # 2^32
            return _uint32(mash_n) * 2.3283064365386963e-10  # 2^-32

        # Initialize state
        self.s0 = mash(" ")
        self.s1 = mash(" ")
        self.s2 = mash(" ")
        self.c = 1

        # Process seed arguments
        for arg in args:
            self.s0 -= mash(arg)
            if self.s0 < 0:
                self.s0 += 1
            self.s1 -= mash(arg)
            if self.s1 < 0:
                self.s1 += 1
            self.s2 -= mash(arg)
            if self.s2 < 0:
                self.s2 += 1

    def random(self):
        """Generate next random number in [0, 1)."""
        self.call_count += 1
        t = 2091639 * self.s0 + self.c * 2.3283064365386963e-10  # 2^-32
        self.s0 = self.s1
        self.s1 = self.s2
        self.c = int(t)
        self.s2 = t - self.c
        return self.s2

    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[int(self.random() * len(seq))]
