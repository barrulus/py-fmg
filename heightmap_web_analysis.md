# Heightmap Web Analysis

## Key Insights from Azgaar's Blog Posts

### Critical Discovery: Height Range and Algorithm Differences

Based on the web research, I've identified a fundamental discrepancy in our understanding:

## 1. Azgaar's Original Heightmap Algorithm (2017)

**From https://azgaar.wordpress.com/2017/04/01/heightmap/:**

### Original Algorithm Parameters:
- **Height Range:** 0-1 (floating point, not 0-255 integers!)
- **Height Calculation:** `newHeight = parentHeight * radius * randomModifier`
- **Random Modifier:** Varies between 0.9-1.1
- **Propagation Threshold:** Stops when height < 0.01
- **Target Polygon Count:** ~8000 polygons

### Key Technical Details:
1. **Queue-based spreading:** Uses BFS to propagate height values
2. **Multiplicative decay:** Heights decay multiplicatively, not additively
3. **Randomness factor:** 0.9-1.1 modifier creates irregularity
4. **Multiple blob support:** Layers different sized "blobs"
5. **No noise functions:** Deliberately avoids Perlin/Simplex noise

## 2. Comparison with Current FMG Implementation

**Current FMG (from our analysis):**
- **Height Range:** 0-100 (uint8 integers)
- **Height Calculation:** `change[q] ** blobPower * (Math.random() * 0.2 + 0.9)`
- **Random Modifier:** 0.9-1.1 (matches original!)
- **Blob Power:** 0.98 for 10000 cells
- **Target Cell Count:** ~10000 cells

## 3. Critical Algorithmic Evolution

The current FMG implementation appears to have evolved significantly from the 2017 blog post:

### Original (2017) vs Current (2024):
| Aspect | Original | Current |
|--------|----------|---------|
| Height Range | 0-1 float | 0-100 uint8 |
| Decay Formula | `parent * radius * random` | `parent ** power * random` |
| Random Range | 0.9-1.1 | 0.9-1.1 |
| Polygon Count | ~8000 | ~10000 |

## 4. Implications for Our Implementation

### Potential Issues:
1. **Height Scaling:** Our 0-100 range might need different scaling
2. **Power vs Multiplicative Decay:** Current FMG uses exponential decay (`**`)
3. **Integer vs Float Precision:** uint8 clamping may affect results

### Our Current Implementation Analysis:
```python
# Our current implementation (matches current FMG):
new_height = (current_height ** self.blob_power) * (self._random() * 0.2 + 0.9)
# blob_power = 0.98 for 10000 cells
```

This matches the **current** FMG implementation, not the original 2017 algorithm.

## 5. Why We're Getting 38.2% vs 78.5% Land

The remaining gap suggests subtle differences in:

### A. Template Processing
The lowIsland template may have evolved since 2017. Our implementation follows the current template syntax but might miss edge cases.

### B. Grid Cell Mapping
The `findGridCell()` function may have nuances we haven't captured.

### C. PRNG Consumption Patterns
Different algorithms might consume random numbers in different orders, causing divergence.

### D. Post-Processing Steps
FMG may have additional processing steps (smoothing, clamping, etc.) after template execution.

## 6. Recommendations for 99.999% Compatibility

### Immediate Actions:
1. **Verify Template Evolution:** Check if lowIsland template has changed since our analysis
2. **Debug PRNG Consumption:** Compare random number consumption patterns
3. **Investigate Grid Mapping:** Ensure `findGridCell()` exact compatibility
4. **Check Post-Processing:** Look for additional steps after heightmap generation

### Deep Analysis Needed:
1. **Runtime Debugging:** Add logging to compare intermediate values
2. **Step-by-Step Comparison:** Compare each template command's output
3. **Floating Point Analysis:** Check for JavaScript vs Python numerical differences

## Conclusion

Our implementation correctly follows the current FMG architecture (exponential decay with blob power), but the 38.2% vs 78.5% land gap indicates remaining subtle algorithmic differences. The web research confirms our approach is fundamentally correct, suggesting the issue lies in implementation details rather than algorithmic understanding.

The next step should be detailed debugging of the lowIsland template execution to identify where our values diverge from FMG's expected output.