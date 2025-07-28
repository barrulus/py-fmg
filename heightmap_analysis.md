# Heightmap Generation Analysis

## Overview
This document provides an extremely granular analysis of each step in FMG's `heightmap-generator.js` to identify why our Python implementation produces different results.

## Key Differences Found

### 1. `getLinePower()` Function Bug
**FMG Original (Line 109-127):**
```javascript
function getLinePower() {
  const linePowerMap = {
    1000: 0.75, 2000: 0.77, 5000: 0.79, 10000: 0.81,
    20000: 0.82, 30000: 0.83, 40000: 0.84, 50000: 0.86,
    60000: 0.87, 70000: 0.88, 80000: 0.91, 90000: 0.92, 100000: 0.93
  };
  return linePowerMap[cells] || 0.81;  // BUG: 'cells' is undefined!
}
```

**Critical Bug:** The function references undefined variable `cells`, causing it to return `undefined`, which when used in math operations defaults to `NaN`, and then gets handled downstream as `1.0`.

**Our Python Implementation:**
```python
def _get_line_power(self, cells: int) -> float:
    # Simulate the bug: undefined variable results in NaN, defaults to 1.0
    return 1.0
```
✅ **Status:** Correctly implemented

### 2. Height Array Initialization
**FMG Original (Line 11):**
```javascript
heights = cells.h ? Uint8Array.from(cells.h) : createTypedArray({maxValue: 100, length: points.length});
```

**Analysis:** 
- FMG uses `Uint8Array` which clamps values to 0-255 range
- Initial values are 0 (all water)
- `createTypedArray` creates array of zeros with specified length

**Our Python Implementation:**
```python
self.heights = np.zeros(self.n_cells, dtype=np.uint8)
```
✅ **Status:** Correctly implemented

### 3. `lim()` Function (Height Clamping)
**FMG Original (Referenced throughout):**
```javascript
const lim = h => minmax(h, 0, 100);  // Clamps to 0-100 range
```

**Our Python Implementation:**
```python
def _lim(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    return np.clip(value, 0, 100).astype(np.uint8)
```
✅ **Status:** Correctly implemented

### 4. Hill Generation Algorithm
**FMG Original (Lines 136-162):**
```javascript
function addOneHill() {
  const change = new Uint8Array(heights.length);
  let h = lim(getNumberInRange(height));
  
  // Find starting point
  do {
    const x = getPointInRange(rangeX, graphWidth);
    const y = getPointInRange(rangeY, graphHeight);
    start = findGridCell(x, y, grid);
    limit++;
  } while (heights[start] + h > 90 && limit < 50);

  change[start] = h;
  const queue = [start];
  while (queue.length) {
    const q = queue.shift();
    
    for (const c of grid.cells.c[q]) {
      if (change[c]) continue;
      change[c] = change[q] ** blobPower * (Math.random() * 0.2 + 0.9);
      if (change[c] > 1) queue.push(c);
    }
  }
  
  heights = heights.map((h, i) => lim(h + change[i]));
}
```

**Critical Analysis:**
1. **No Range Constraints in Spreading:** FMG does NOT check if neighbors are within the range constraints during blob spreading
2. **Global Application:** The `change` array is applied to ALL cells at the end
3. **Power Decay:** Uses `change[q] ** blobPower` for decay calculation

**Our Python Implementation Issue:**
```python
# Skip neighbors outside the specified range (this is the key fix!)
if not (x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max):
    continue
```

❌ **MAJOR BUG FOUND:** Our implementation incorrectly constrains blob spreading to range boundaries. FMG does NOT do this constraint check during spreading - it only uses the range to find the initial starting point.

### 5. `findGridCell()` Function
**FMG Usage (Line 145):**
```javascript
start = findGridCell(x, y, grid);
```

**Analysis:** This function maps continuous x,y coordinates to discrete grid cell indices. The implementation depends on the grid structure.

**Our Python Implementation:**
```python
def _find_grid_cell(self, x: float, y: float) -> int:
    col = min(int(x / self.config.spacing), self.config.cells_x - 1)
    row = min(int(y / self.config.spacing), self.config.cells_y - 1)
    return row * self.config.cells_x + col
```
⚠️ **Potential Issue:** May not match FMG's exact grid mapping algorithm.

### 6. `getPointInRange()` Function
**FMG Original (Lines 508-517):**
```javascript
function getPointInRange(range, length) {
  if (typeof range !== "string") {
    ERROR && console.error("Range should be a string");
    return;
  }
  
  const min = range.split("-")[0] / 100 || 0;
  const max = range.split("-")[1] / 100 || min;
  return rand(min * length, max * length);
}
```

**Analysis:**
- Splits range string like "60-80" into percentages (0.6-0.8)
- Multiplies by length to get absolute coordinates
- Uses `rand()` function for random value in range

**Our Python Implementation:**
```python
def _get_point_in_range(self, range_str: str, max_val: float) -> float:
    if '-' in range_str:
        min_pct, max_pct = map(float, range_str.split('-'))
        min_val = max_val * min_pct / 100
        max_val_range = max_val * max_pct / 100
        return min_val + self._random() * (max_val_range - min_val)
    
    pct = float(range_str)
    return max_val * pct / 100
```
✅ **Status:** Logic matches, but random number generator may differ

### 7. Random Number Generation
**FMG Original (Line 68):**
```javascript
Math.random = aleaPRNG(seed);
```

**Analysis:** FMG overrides `Math.random` with Alea PRNG seeded with specific seed value.

**Our Python Implementation:**
```python
from py_fmg.utils.random import set_random_seed, get_prng
```

⚠️ **Potential Issue:** Different PRNG implementation may produce different sequences even with same seed.

### 8. Graph Structure Access
**FMG Original (Line 154):**
```javascript
for (const c of grid.cells.c[q]) {
```

**Analysis:** Accesses cell neighbors through `grid.cells.c[q]` array.

**Our Python Implementation:**
```python
for neighbor in self.graph.cell_neighbors[current]:
```
✅ **Status:** Equivalent neighbor access pattern

## Root Cause Analysis

### Primary Issue: Incorrect Range Constraint Implementation
The most critical difference is in hill generation blob spreading:

**FMG Behavior:**
1. Uses range constraints ONLY to find starting point
2. Blob spreading occurs without range constraints
3. Can spread beyond the initial range boundaries
4. This allows hills to have wider influence

**Our Behavior:**
1. Uses range constraints to find starting point ✅
2. INCORRECTLY constrains blob spreading to range boundaries ❌
3. Severely limits hill influence
4. Results in much lower overall heights

### Secondary Issues:
1. **Grid Cell Mapping:** Our `_find_grid_cell()` may not exactly match FMG's implementation
2. **Random Number Generation:** Different PRNG sequences despite same seed
3. **Coordinate System:** Potential differences in how x,y coordinates map to grid indices

## Recommended Fixes

### 1. Remove Range Constraints from Blob Spreading (CRITICAL)
```python
# In _add_one_hill(), remove this check:
# if not (x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max):
#     continue

# Allow unrestricted spreading like FMG
for neighbor in self.graph.cell_neighbors[current]:
    if neighbor not in visited:
        visited.add(neighbor)
        # Calculate new height with power decay and randomness
        new_height = (current_height ** self.blob_power) * (self._random() * 0.2 + 0.9)
        
        if new_height > 1:
            change[neighbor] = new_height
            queue.append(neighbor)
```

### 2. Investigate Grid Cell Mapping
Compare our `_find_grid_cell()` implementation with FMG's `findGridCell()` function to ensure identical coordinate-to-cell mapping.

### 3. Verify Random Number Generation
Ensure our Alea PRNG implementation produces identical sequences to FMG's JavaScript version.

## Expected Impact
Removing the incorrect range constraints should dramatically increase the height values and land percentage, bringing our results closer to FMG's expected output of:
- Height range: 0-51 (vs our current 0-23)
- Mean height: 26.6 (vs our current 2.7)  
- Land percentage: 78.5% (vs our current ~0%)

This single fix should resolve the primary scaling issue in our heightmap generation.