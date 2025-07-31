# Status Report: FMG Heightmap Generation Port

  Executive Summary

  We have made significant progress debugging the isthmus
   template's PRNG desynchronization issue. Through
  careful analysis and implementation of FMG's exact blob
   spreading logic (including its quirks), we reduced
  PRNG calls from ~595,846 to a more reasonable range.
  However, this fix has revealed that other templates
  (volcano, peninsula, lowIsland, highIsland) now produce
   incorrect results, indicating deeper systemic issues
  with our heightmap generation implementation.

  Key Accomplishments

  1. Identified and Fixed Critical PRNG Function Usage
    - Implemented three distinct random functions:
  _rand(), _P(), and _random()
    - Fixed _get_number_in_range() to handle decimals
  with P()
    - Fixed _get_point_in_range() to use _rand() for
  integer ranges
    - Fixed add_strait and invert to use _P() for
  probability checks
  2. Discovered and Replicated FMG's Blob Spreading
  Quirks
    - FMG uses Uint8Array for the change array, causing
  automatic float-to-int truncation
    - This creates "stalled decay" where certain height
  values produce flat-topped plateaus
    - The critical fix: check the STORED truncated value
  (if change[neighbor] > 1) after assignment, not the
  float value before
  3. Implemented All Analyst Recommendations
    - ✅ Fixed _get_line_power() to use dictionary lookup
   instead of hardcoded 0.81
    - ✅ Added guard clause to _add_one_pit() to prevent
  cell re-processing
    - ✅ Verified _add_one_hill() guard clause
  implementation

  Current Status

  PRNG Call Counts for Isthmus Template:
  - Initial: ~595,846 calls (far too many)
  - After fixes: ~12,143 calls for first Hill command (8
  hills)
  - Each hill now uses 1,000-2,800 PRNG calls (down from
  18,000)
  - Still higher than FMG's reported ~320 total calls

  Spreading Behavior:
  - Hills: Create large plateaus affecting ~3,361 cells
  (was 10,010)
  - Pits: 200-500 PRNG calls, affect 100-300 cells
  - Troughs: 300-700 PRNG calls per trough

  Remaining Issues

  1. Other Templates Now Broken
    - volcano, peninsula, lowIsland, highIsland templates
   produce incorrect results
    - Suggests our fix may be too specific to isthmus or
  there are template-specific behaviors
  2. PRNG Count Still Too High
    - Even with improvements, we use significantly more
  PRNG calls than FMG
    - Possible causes:
        - Different grid connectivity structure
      - Missing early termination logic
      - Fundamental algorithm differences we haven't
  identified
  3. Need for Comprehensive Testing
    - Require exact heightmap array comparison with FMG
    - Need to test all templates, not just isthmus

  Technical Insights

  The core issue was understanding FMG's two distinct
  spreading mechanisms:

  1. Integer Truncation (Hills):
  change[c] = change[q] ** blobPower * (Math.random() *
  0.2 + 0.9);
    - Assigns float to Uint8Array, causing truncation
    - Creates decay stalling and large plateaus
  2. Smooth Float Decay (Pits/Troughs):
  h = h ** blobPower * (Math.random() * 0.2 + 0.9);
    - Maintains float precision
    - Natural termination when h < 1

  Next Steps

  1. Investigate Template-Specific Issues
    - Test each broken template individually
    - Identify what makes them fail with current
  implementation
  2. Create Comprehensive Test Suite
    - Implement heightmap array comparison as analyst
  suggested
    - Test all templates with known seeds
  3. Further PRNG Optimization
    - Investigate why even improved counts are higher
  than FMG
    - Consider grid connectivity differences
  4. Document Template Behaviors
    - Map out which templates use which algorithms
    - Identify any template-specific quirks

  The project has made substantial progress in
  understanding FMG's complex heightmap generation, but
  full compatibility remains elusive due to the intricate
   interplay of algorithms and their various edge cases.
