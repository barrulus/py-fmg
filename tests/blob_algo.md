Of course. This is an excellent question because FMG's "blob spreading" is one of the most fundamental yet non-obvious algorithms in its entire generation pipeline. Its behavior is an emergent property of several interacting systems, not just a simple mathematical formula.

Here is an extremely detailed explanation of how it works, what the math is, and how various factors influence it, based on the `addHill` function in `heightmap-generator.txt`.

### 1. High-Level Overview: What is Blob Spreading?

At its core, FMG's blob spreading is a procedural brush used to raise or lower terrain in an organic, blob-like shape. It's not a simple circle or square brush; it grows outwards from a starting point, creating irregular coastlines and landmasses.

The algorithm is a modified **Breadth-First Search (BFS)** that operates on the Voronoi graph. It starts with a high "energy" or "height value" at a central cell and propagates that energy outwards to its neighbors, decaying it at each step until it fades to nothing.

---

### 2. The Core Algorithm: A Step-by-Step Walkthrough (`addHill`)

The canonical example is the `addHill` function. Let's trace its execution precisely.

#### Step A: Initialization

1.  **Create a Temporary `change` Map:** A new, temporary array called `change` is created. It's the same size as the main `heights` array and is initialized to all zeros. Crucially, it is a `new Uint8Array()`. This data type is essential to the algorithm's behavior.
2.  **Determine Initial Height (`h`):** The function gets an initial integer height value `h` (e.g., `30`) based on the template command (`Hill 1 20-30 ...`).
3.  **Find a Start Point:** A random starting cell index (`start`) is chosen within the template's constraints (e.g., within a certain X/Y range).
4.  **"Seed" the BFS:** The initial height `h` is placed into the temporary map at the starting location: `change[start] = h;`. A `queue` is created for the BFS, and the `start` cell is its first and only item.

#### Step B: The Spreading Loop (The BFS)

The code now enters a `while (queue.length)` loop. As long as there are cells in the queue to be processed, the spread continues.

1.  **Dequeue:** The algorithm takes the first cell (`current`) out of the `queue`.
2.  **Iterate Neighbors:** It then gets the list of all neighbors of the `current` cell from the Voronoi graph (`grid.cells.c[current]`).
3.  **For Each Neighbor:** It performs the following sequence for every single neighbor:
    a. **The Guard Clause:** It first checks `if (change[neighbor])`. In JavaScript, this evaluates to `true` if the value is anything other than `0`. This is the **most important safety check**. It ensures that a cell that has already been processed (i.e., had a height value written to it) will never be processed again. This prevents infinite loops and ensures the wave only moves outwards.
    b. **The Decay Calculation:** If the guard clause passes (meaning the neighbor is "new territory"), it calculates the height for this neighbor. This is the core mathematical formula.
    c. **The Termination Condition & Assignment:** It then checks if the calculated height is significant enough to continue the spread. If it is, the value is stored in `change[neighbor]` and the `neighbor` is added to the back of the `queue`.

#### Step C: Final Application

Once the `while` loop finishes (meaning the `queue` is empty and the spread has died out), the temporary `change` array, which now contains the full blob shape, is added to the main `heights` array to modify the final terrain.

---

### 3. The Mathematics of Decay

The single most important line is the decay formula:

```javascript
change[neighbor] = change[current] ** blobPower * (Math.random() * 0.2 + 0.9);
```

Let's break this down:

*   **`change[current]`**: This is the height value of the cell we are currently spreading *from*. This is an **integer** because it was read from the `Uint8Array`.
*   **`** blobPower`**: This is the primary decay mechanism. `blobPower` is a value very close to 1 (e.g., `0.98` for a 10k cell map). Raising the current height to this power reduces it slightly.
    *   `30 ** 0.98` ≈ `28.03`
    *   `10 ** 0.98` ≈ `9.55`
    *   The closer `blobPower` is to 1, the slower the decay, and the farther the blob will spread. Think of it as viscosity: a low `blobPower` (e.g., `0.8`) is like water (spreads little), while a high `blobPower` (e.g., `0.99`) is like thick syrup (spreads far).
*   **`(Math.random() * 0.2 + 0.9)`**: This is the random noise factor. It generates a random float between `0.9` and `1.1`. Its purpose is to make the blob's edges irregular and organic. Without this, the blob would spread out in a near-perfectly symmetrical shape. Multiplying the decayed height by this factor can either slightly speed up the decay (if the random number is `< 1.0`) or slightly slow it down (if `> 1.0`).

---

### 4. The "Secret Sauce": How Interacting Systems Create the Final Behavior

The math alone doesn't tell the whole story. The final behavior is an emergent property of three interacting systems.

#### A. The Role of `Uint8Array` (Integer Truncation)

This is the most critical and non-obvious component. When the floating-point result of the decay calculation is assigned back to the `change` array, it is **automatically truncated to an integer**.

*   Calculation: `28.03 * 1.05` = `29.4315`
*   Assignment: `change[neighbor] = 29.4315;`
*   **Result Stored in Array:** The `Uint8Array` stores the value `29`.

This truncation is the source of the "plateau" effect you debugged. When the height value gets low, the decay can stall. For example, if `change[current]` is `4`, the calculation `4**0.98 * 1.05` results in `~4.04`, which is stored as `4`. The height value hasn't decayed at all.

#### B. The Role of the Guard Clause (`if (change[neighbor])`)

Because the decay can stall, the guard clause is the **only thing preventing a runaway loop**. Even if the height value `4` propagates outwards without changing, the BFS can only visit each cell **once**. The guard clause ensures that once a cell has a value of `4`, it can never be added to the queue again. This forces the spread to terminate when it runs out of fresh, zero-value cells to conquer.

#### C. The Role of the Termination Condition (`if (change[c] > 1)`)

This is the final key to the puzzle, as you discovered. FMG's code has a specific order of operations: **calculate, store, then check the stored value.**

1.  `new_height_float` is calculated (e.g., `1.87`).
2.  `change[neighbor]` is assigned `1.87`, which is stored as `1`.
3.  The check is `if (change[neighbor] > 1)`, which becomes `if (1 > 1)`. This is **FALSE**.

This logic is what prevents the "wave of 1s." The moment the stored, truncated integer value drops to `1`, the spread is guaranteed to stop in that direction.

### 5. Summary: Factors That Affect the Spread

*   **Initial `h` (Height):** A higher starting value means the spread has more "energy" and will travel farther before decaying to `1`.
*   **`blobPower`:** The single most important factor for spread distance. It's derived from `cellsDesired`. More cells = higher `blobPower` = wider blobs to ensure landmasses are an appropriate relative size.
*   **Graph Topology:** In areas where cells have more neighbors (e.g., 7 or 8), the blob will spread faster and wider than in areas where cells have fewer neighbors (e.g., 5).
*   **Randomness:** The `* (0.9 to 1.1)` factor primarily affects the *shape* and *irregularity* of the blob's edge. On average, its effect on spread distance is minimal, but it can cause lobes and inlets to form.

