"""
FMG Core Generation Flow Analysis

Based on main.js:616-698, the FMG generation follows this sequence:

1. Grid Generation (lines 628-629) ✅
   - generateGrid() or use precreated graph
   - Creates Voronoi cell structure
   - **Python Status**: ✅ Complete with all FMG features
   - Lloyd's relaxation, height pre-allocation, grid reuse

2. Heightmap Generation (line 630) ✅
   - HeightmapGenerator.generate(grid)
   - Creates elevation data for all cells
   - **Python Status**: ✅ All algorithms implemented
   - Template support, all terrain features working

3. Geographic Features (lines 633-636)
   - Features.markupGrid() - basic geographic classification
   - addLakesInDeepDepressions() - lake detection in depressions
   - openNearSeaLakes() - connect lakes to sea where appropriate

4. Ocean and Map Setup (lines 637-639)
   - OceanLayers() - ocean depth and currents
   - defineMapSize() - set map dimensions
   - calculateMapCoordinates() - coordinate system

5. Climate Generation (lines 640-641)
   - calculateTemperatures() - temperature based on latitude/elevation
   - generatePrecipitation() - wind/moisture simulation

6. Graph Reprocessing (lines 643-645) ✅
   - reGraph() - rebuild graph with new data
   - Features.markupPack() - mark geographic features on pack
   - createDefaultRuler() - measurement tools
   - **Python Status**: ✅ reGraph/cell packing complete
   - Full implementation with coastal enhancement

7. Hydrology (lines 647-648)
   - Rivers.generate() - river flow simulation
   - Biomes.define() - biome assignment based on climate

8. Cultural/Political Systems (lines 650-660)
   - rankCells() - cell suitability scoring
   - Cultures.generate() & expand() - cultural regions
   - BurgsAndStates.generate() - cities and political boundaries
   - Routes.generate() - trade routes
   - Religions.generate() - religious systems
   - BurgsAndStates.defineStateForms() - government types
   - Provinces.generate() & getPoles() - administrative divisions
   - BurgsAndStates.defineBurgFeatures() - city characteristics

9. Finalization (lines 661-667)
   - Rivers.specify() - river details (width, etc.)
   - Features.specify() - geographic feature details
   - Military.generate() - military units
   - Markers.generate() - map markers
   - Zones.generate() - special zones

10. Rendering (lines 668-669)
    - drawScaleBar() - scale bar
    - Names.getMapName() - map title

Key Dependencies:
- Each stage builds on previous ones
- Geographic features depend on heightmap
- Climate depends on geography
- Biomes depend on climate
- Political systems depend on geography + climate + biomes
- Rivers interact with multiple stages (heightmap modification, flow calculation)

Critical Data Structures:
- grid: Voronoi cell graph with neighbors, heights, features
- pack: Processed geographic and political data
- cells: Individual cell properties (height, biome, culture, etc.)
"""