# py-fmg — File‑by‑File Porting Roadmap (GeoJSON‑First, Leaflet Previews)

> This roadmap maps **/Fantasy-Map-Generator** JavaScript modules to their **py-fmg** Python counterparts, the **GeoJSON artifacts** they must emit, the **Leaflet previews** to generate, and the **tests** to write. Order is dependency‑aware. If a JS filename differs locally, keep the intent and slot it into the same step.

---

## Conventions

- **JS path:** `/Fantasy-Map-Generator/modules/<file>.js`
- **Python target:** `py_fmg/<module>.py` (one JS → one Py)
- **GeoJSON out:** `out/geojson/<stage>/<layer>.geojson`
- **Preview:** `out/preview/<stage>.html` (Leaflet)
- **Tests:** `tests/test_<module>.py` with seeded snapshots
- **Importers:** `importers/<layer>_importer.py` (shared: `importers/common.py`)
- **Every feature** carries `properties.map_id` and stage‑specific attributes

---

## 0) Bootstrap

1. **Repo & Config**

   - Py: `py_fmg/config.py`, `py_fmg/paths.py`, `py_fmg/logging.py`
   - Tests: `tests/test_config.py`
   - Output: none
   - Notes: Centralize `seed`, sizes, CRS (EPSG:4326), out dirs.

2. **RNG & Utils**

   - JS: `utils.js`, `seed.js` (or RNG helpers)
   - Py: `py_fmg/utils.py`, `py_fmg/rng.py`
   - Tests: `tests/test_utils.py`, `tests/test_rng.py` (JS‑parity fixtures)
   - Output: none

3. **CLI & Manifests**

   - Py: `cli/main.py` (Typer/Click), `py_fmg/manifest.py`
   - Tests: `tests/test_manifest.py`
   - Output: `out/geojson/<map_id>/manifest.json`

---

## 1) Mesh / Cells / Neighbors

4. **Points & Mesh Seed**

   - JS: `points.js`
   - Py: `py_fmg/points.py`
   - GeoJSON: none (feeds cells)
   - Preview: none
   - Tests: first‑N point snapshot under fixed seed

5. **Voronoi/Delaunay & Cells**

   - JS: `grid.js` / `mesh.js`
   - Py: `py_fmg/mesh.py`
   - GeoJSON: `cells.geojson` (Polygon); properties: `cell_id`, `centroid`, `neighbors` (array of ids)
   - Preview: `cells.html` (thin polygon outlines)
   - Tests: cell count, centroid snapshots, neighbor degree histogram parity

6. **Lloyd Relaxation (if present)**

   - JS: `relaxation.js`
   - Py: `py_fmg/relax.py`
   - GeoJSON: re‑emit `cells.geojson` if relaxation modifies geometry
   - Preview: `cells_relaxed.html`
   - Tests: variance reduction of cell area; deterministic centers

---

## 2) Heightmap & Terrain

7. **Heightmap Generation**

   - JS: `heightmap.js`
   - Py: `py_fmg/heightmap.py`
   - GeoJSON: `cells_height.geojson` (same polygons, props: `elevation` in \[0,1])
   - Preview: `heightmap.html` (choropleth by elevation quantiles)
   - Tests: min/max/percentiles parity; first‑N elevations snapshot

8. **Erosion/Smoothing (if separate)**

   - JS: `erosion.js` or embedded
   - Py: `py_fmg/erosion.py`
   - GeoJSON: `cells_height.geojson` (updated)
   - Preview: `heightmap_erosion.html`
   - Tests: relief stats change as expected; snapshot of modified elevations

---

## 3) Land/Water Masks & Coastlines

9. **Ocean/Land/Lakes Mask**

   - JS: `ocean.js` / `land-water.js`
   - Py: `py_fmg/watermask.py`
   - GeoJSON: `cells_watermask.geojson` (props: `is_ocean`, `is_coast`, `is_lake`)
   - Preview: `coast.html` (blue water, tan land)
   - Tests: ocean coverage %, lake count, coast cell count parity

10. **Coastline Extraction**

    - JS: `coastline.js`
    - Py: `py_fmg/coastline.py`
    - GeoJSON: `coastlines.geojson` (MultiLineString)
    - Preview: `coastlines.html`
    - Tests: total coastline length ≈ JS; first‑N segment endpoints snapshot

---

## 4) Climate & Biomes

11. **Temperature**

    - JS: `temperature.js`
    - Py: `py_fmg/temperature.py`
    - GeoJSON: `cells_temperature.geojson` (props: `temperature`)
    - Preview: `temperature.html` (choropleth)
    - Tests: distribution & latitudinal gradient parity

12. **Precipitation**

    - JS: `precipitation.js`
    - Py: `py_fmg/precipitation.py`
    - GeoJSON: `cells_precip.geojson` (props: `precip`)
    - Preview: `precipitation.html`
    - Tests: histogram parity; correlation with elevation where expected

13. **Biomes**

    - JS: `biomes.js`
    - Py: `py_fmg/biomes.py`
    - GeoJSON: `cells_biomes.geojson` (props: `biome_code`, `biome_name`)
    - Preview: `biomes.html` (categorical legend)
    - Tests: biome histogram parity; threshold edge cases

---

## 5) Hydrology

14. **Flow Direction & Accumulation**

    - JS: `hydrology.js` (or inside rivers)
    - Py: `py_fmg/hydrology.py`
    - GeoJSON: `cells_flow.geojson` (props: `flow_dir`, `flow_acc`)
    - Preview: `flow.html` (small arrows or color for accumulation)
    - Tests: sink handling; top‑k accumulation cells snapshot

15. **Rivers**

    - JS: `rivers.js`
    - Py: `py_fmg/rivers.py`
    - GeoJSON: `rivers.geojson` (LineString; props: `river_id`, `order`, `discharge`, `length`)
    - Preview: `rivers.html` (line weight by order/discharge)
    - Tests: count, total length, mouth locations parity

---

## 6) Cultures, Names, Burgs

16. **Cultures**

    - JS: `cultures.js`
    - Py: `py_fmg/cultures.py`
    - GeoJSON: `cultures.geojson` (Polygon or attributes per cell depending on FMG; include `culture_id`, `name`)
    - Preview: `cultures.html` (categorical fills)
    - Tests: count, area coverage, seed determinism

17. **Names Base & Generator**

    - JS: `names-base.js`, `names-generator.js`
    - Py: `py_fmg/names.py`
    - GeoJSON: none (used by burgs/states)
    - Preview: none
    - Tests: deterministic sequences; diacritic handling

18. **Burgs & Castles**

    - JS: `burgs-and-castles.js`
    - Py: `py_fmg/burgs.py`
    - GeoJSON: `burgs.geojson` (Point; props: `burg_id`, `name`, `population`, `is_port`, `culture_id`)
    - Preview: `burgs.html` (markers + name label)
    - Tests: count distribution, coastal port validation, nearest‑coast distance checks

---

## 7) States, Provinces, Borders

19. **States & Provinces Seeder**

    - JS: `states-and-provinces.js`
    - Py: `py_fmg/states.py`
    - GeoJSON: `states.geojson` (Polygon; props: `state_id`, `name`, `capital_burg_id`)
    - Preview: `states.html` (categorical fills + capital markers)
    - Tests: count/area parity; capitals exist and lie within state polygon

20. **Provinces Generator**

    - JS: `provinces-generator.js`
    - Py: `py_fmg/provinces.py`
    - GeoJSON: `provinces.geojson` (Polygon; props: `province_id`, `state_id`, `name`)
    - Preview: `provinces.html` (categorical fills + state borders overlay)
    - Tests: province count, hierarchy integrity (province ⊂ state)

21. **Borders**

    - JS: `borders.js`
    - Py: `py_fmg/borders.py`
    - GeoJSON: `borders.geojson` (MultiLineString; props: `type` ∈ {state, province})
    - Preview: `borders.html`
    - Tests: border continuity; shared edge equality within tolerance

---

## 8) Routes (Roads & Maritime)

22. **Roads / Land Routes**

    - JS: `roads.js` / `routes.js`
    - Py: `py_fmg/routes.py`
    - GeoJSON: `routes.geojson` (LineString; props: `route_id`, `class`, `distance`, `from_burg`, `to_burg`)
    - Preview: `routes.html` (dash arrays for class; snaps to burgs)
    - Tests: connected components of burg graph; average/median route length parity

23. **Sea Routes / Trade**

    - JS: `trade.js` (if maritime inside)
    - Py: `py_fmg/trade.py`
    - GeoJSON: `sea_routes.geojson` (LineString; props: `from_port`, `to_port`, `distance`)
    - Preview: `sea_routes.html`
    - Tests: port validation; no lines through land (intersect‑free vs land mask)

---

## 9) Regiments & Markers

24. **Regiments**

    - JS: `regiments.js` (if present)/game overlays
    - Py: `py_fmg/regiments.py`
    - GeoJSON: `regiments.geojson` (Point/Line depending on logic; props: `type`, `size`, `owner`)
    - Preview: `military.html`
    - Tests: schema & placement rules (e.g., within state)

25. **Markers / Labels / Grid**

    - JS: `markers.js`, `labels.js`, `grid-overlays.js`
    - Py: `py_fmg/markers.py`, `py_fmg/labels.py`, `py_fmg/grids.py`
    - GeoJSON: `markers.geojson` (Point; props: `marker_type`, `text`)
    - Preview: `markers.html`
    - Tests: label collision score; marker types whitelist

---

## 10) Exporters (GeoJSON) & Importers (PostGIS)

26. **GeoJSON Exporter**

    - Py: `py_fmg/exporter.py`
    - Output: writes all FeatureCollections per stage + per‑map `manifest.json`
    - Tests: schema validation; stable ordering for diffable artifacts

27. **PostGIS Importers**

    - Py: `importers/common.py`, `importers/cells.py`, `importers/rivers.py`, `importers/routes.py`, `importers/burgs.py`, `importers/states.py`, `importers/provinces.py`, `importers/regiments.py`, `importers/markers.py`
    - CLI: `py-fmg import-all --manifest out/geojson/<map_id>/manifest.json`
    - Tests: ingest into test DB; geometry validity (`make_valid`), SRID checks; FK integrity

---

## 11) Leaflet Preview System

28. **Preview Template**

    - Py/HTML: `py_fmg/preview/leaflet_template.py` → produces HTML with toggles
    - Output: `out/preview/<stage>.html`, `out/preview/full_map.html`
    - Tests: smoke test (file exists, contains layer links)

---

## 12) API (Optional in Parity Phase)

29. **FastAPI Surface**

    - Py: `app/main.py`
    - Endpoints: `POST /maps/generate`, `GET /maps/{map_id}/manifest`, `POST /import`
    - Tests: minimal integration against temp DB

---

## 13) Documentation & Ops

30. **Docs & Status**

    - Docs: `README.md` (GeoJSON‑first), `AGENTS.md`, `TASKS.md`, `docs/porting-status.md`
    - CI: GitHub Actions (pytest, artifact upload of sample GeoJSON & previews)

---

## Deliverables per File (Checklist)

- Python module created
- Seeded tests + snapshots
- GeoJSON emitted to `out/geojson/...`
- Leaflet preview HTML generated
- Importers verified (as applicable)
- Notes added to `docs/porting-status.md` (parity, deviations, TODOs)
