#!/usr/bin/env python3
"""
Test the JavaScript rand() function behavior.
"""

def test_js_rand():
    """Test JavaScript rand() function behavior."""
    
    print("🔍 JAVASCRIPT RAND() FUNCTION TEST")
    print("=" * 50)
    
    print("\nJS rand() implementation:")
    print("```javascript")
    print("function rand(min, max) {")
    print("  if (min === undefined && max === undefined) return Math.random();")
    print("  if (max === undefined) {")
    print("    max = min;")
    print("    min = 0;")
    print("  }")
    print("  return Math.floor(Math.random() * (max - min + 1)) + min;")
    print("}")
    print("```")
    
    print("\n📊 TESTING BEHAVIOR:")
    print("-" * 40)
    
    # Test cases
    test_cases = [
        (0, 10, "rand(0, 10)"),
        (5, 15, "rand(5, 15)"),
        (90, 99, "rand(90, 99)"),
        (180, 240, "rand(180, 240)"),  # 60-80% of 300
        (135, 165, "rand(135, 165)"),  # 45-55% of 300
    ]
    
    print("\nFor various ranges:")
    for min_val, max_val, desc in test_cases:
        # JS formula: Math.floor(Math.random() * (max - min + 1)) + min
        # With random=0: floor(0 * (max-min+1)) + min = min
        # With random=0.999: floor(0.999 * (max-min+1)) + min ≈ max
        range_size = max_val - min_val + 1
        print(f"\n{desc}:")
        print(f"  Range: {min_val} to {max_val}")
        print(f"  Possible values: {range_size} integers")
        print(f"  With random=0.0: {min_val}")
        print(f"  With random=0.5: {int(0.5 * range_size) + min_val}")
        print(f"  With random=0.999: {int(0.999 * range_size) + min_val}")
    
    print("\n💡 KEY INSIGHT:")
    print("-" * 40)
    print("JavaScript rand() returns INTEGERS in range [min, max] inclusive")
    print("Our Python was returning FLOATS, causing different cell selections!")
    
    print("\n🔍 COORDINATE EXAMPLE:")
    print("-" * 40)
    print("For range '60-80' on 300px canvas:")
    print("  JS: rand(180, 240) → integers 180-240")
    print("  PY (old): 180 + random() * 60 → floats 180.0-240.0")
    print("  PY (new): int(180 + random() * 61) → integers 180-240")
    
    print("\nThis difference means:")
    print("- Different cells selected for hill placement")
    print("- Different spreading patterns")
    print("- Different final terrain distribution")

if __name__ == "__main__":
    test_js_rand()