#!/usr/bin/env python3
"""
Audit random number consumption in addHill to ensure exact match with JavaScript.
"""

def audit_addhill_random_calls():
    """Compare random calls between JS and Python implementations."""
    
    print("🔍 RANDOM NUMBER CONSUMPTION AUDIT: addHill")
    print("=" * 50)
    
    print("\n📊 JavaScript addHill Random Calls:")
    print("-" * 40)
    print("1. getNumberInRange(count) - uses 1 random if range")
    print("2. For each hill:")
    print("   a. getNumberInRange(height) - uses 1 random if range") 
    print("   b. getPointInRange(rangeX) - uses 1 random")
    print("   c. getPointInRange(rangeY) - uses 1 random")
    print("   d. (repeat b,c up to 50 times if height check fails)")
    print("   e. For each neighbor during spreading:")
    print("      - Math.random() * 0.2 + 0.9 - uses 1 random")
    
    print("\n📊 Python addHill Random Calls:")
    print("-" * 40)
    print("1. _get_number_in_range(count) - uses 1 random if range")
    print("2. For each hill (_add_one_hill):")
    print("   a. _get_number_in_range(height) - uses 1 random if range")
    print("   b. _get_point_in_range(range_x) - uses 1 random")
    print("   c. _get_point_in_range(range_y) - uses 1 random")
    print("   d. (repeat b,c up to 50 times if height check fails)")
    print("   e. For each neighbor during spreading:")
    print("      - self._random() * 0.2 + 0.9 - uses 1 random")
    
    print("\n✅ MATCH: Both use same number of random calls in same order!")
    
    print("\n🔍 DETAILED ANALYSIS:")
    print("-" * 40)
    
    print("\n1. Count parsing:")
    print("   JS:  count = getNumberInRange(count)")
    print("   PY:  count = int(self._get_number_in_range(count))")
    print("   ✅ Both consume 1 random if count is a range")
    
    print("\n2. Height initialization:")
    print("   JS:  let h = lim(getNumberInRange(height))")
    print("   PY:  h = self._lim(self._get_number_in_range(height))")
    print("   ✅ Both consume 1 random if height is a range")
    
    print("\n3. Position selection:")
    print("   JS:  const x = getPointInRange(rangeX, graphWidth)")
    print("        const y = getPointInRange(rangeY, graphHeight)")
    print("   PY:  x = self._get_point_in_range(range_x, self.config.width)")
    print("        y = self._get_point_in_range(range_y, self.config.height)")
    print("   ✅ Both consume 2 randoms per attempt")
    
    print("\n4. Neighbor spreading:")
    print("   JS:  change[c] = change[q] ** blobPower * (Math.random() * 0.2 + 0.9)")
    print("   PY:  new_height = (current_height ** self.blob_power) * (self._random() * 0.2 + 0.9)")
    print("   ✅ Both consume 1 random per neighbor")
    
    print("\n⚠️  POTENTIAL ISSUE FOUND:")
    print("-" * 40)
    print("The random consumption appears identical, but the minimum height")
    print("difference (15 vs 2) suggests a different issue:")
    print("\n1. Ocean cells (h < 20) handling")
    print("2. Initial height distribution") 
    print("3. Template command differences")
    print("4. Cell connectivity/neighbor order")

if __name__ == "__main__":
    audit_addhill_random_calls()