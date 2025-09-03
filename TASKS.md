# TASKS.md — File-by-File Roadmap for **py-fmg**

This roadmap lists every FMG module (JavaScript) and its Python destination, along with outputs and verification.

---

## 0) Bootstrap

- **New files:**

  - `py_fmg/__init__.py`
  - `py_fmg/config.py`
  - `py_fmg/db.py`
  - `py_fmg/models.py`
  - `cli/main.py`
  - `importers/postgis.py`

- **Outputs:** basic schema + CLI skeleton

---

## 1) Utilities

- JS: `utils.js`, `random.js`
- Py: `py_fmg/utils.py`, `py_fmg/rng.py`
- Outputs: seeded RNG fixtures, math helpers
- Tests: parity with FMG sequences

---

## 2) Mesh & Cells

- JS: `grid.js`, `mesh.js`, `relaxation.js`
- Py: `py_fmg/mesh.py`, `py_fmg/relax.py`
- Outputs: **cells.geojson** (polygons + neighbors)
- Preview: `cells.html`

---

## 3) Heightmap

- JS: `heightmap.js`
- Py: `py_fmg/heightmap.py`
- Outputs: **cells.geojson** with `elevation`
- Preview: `heightmap.html`

---

## 4) Oceans & Coastlines

- JS: `ocean.js`, `coastline.js`
- Py: `py_fmg/watermask.py`, `py_fmg/coastline.py`
- Outputs: **coastlines.geojson**, updated cells
- Preview: `coast.html`

---

## 5) Climate & Biomes

- JS: `temperature.js`, `precipitation.js`, `biomes.js`
- Py: `py_fmg/temperature.py`, `py_fmg/precipitation.py`, `py_fmg/biomes.py`
- Outputs: **cells.geojson** with climate & biome props
- Preview: `biomes.html`

---

## 6) Rivers

- JS: `rivers.js`
- Py: `py_fmg/rivers.py`
- Outputs: **rivers.geojson**
- Preview: `rivers.html`

---

## 7) Cultures, Names, Burgs

- JS: `cultures.js`, `names-base.js`, `names-generator.js`, `burgs-and-castles.js`
- Py: `py_fmg/cultures.py`, `py_fmg/names.py`, `py_fmg/burgs.py`
- Outputs: **burgs.geojson**
- Preview: `burgs.html`

---

## 8) States & Provinces

- JS: `states-and-provinces.js`, `provinces-generator.js`, `borders.js`
- Py: `py_fmg/states.py`, `py_fmg/provinces.py`, `py_fmg/borders.py`
- Outputs: **states.geojson**, **provinces.geojson**, **borders.geojson**
- Preview: `states.html`

---

## 9) Routes & Trade

- JS: `roads.js`, `routes.js`, `trade.js`
- Py: `py_fmg/routes.py`, `py_fmg/trade.py`
- Outputs: **routes.geojson**, **sea_routes.geojson**
- Preview: `routes.html`

---

## 10) Regiments & Markers

- JS: `regiments.js`, `markers.js`
- Py: `py_fmg/regiments.py`, `py_fmg/markers.py`
- Outputs: **regiments.geojson**, **markers.geojson**
- Preview: `military.html`

---

## 11) Exporter

- JS: `export.js`
- Py: `py_fmg/exporter.py`
- Outputs: per-map **manifest.json** listing all GeoJSON layers

---

## 12) Importer

- New in Python only
- Py: `importers/postgis.py`
- Reads GeoJSON into PostGIS tables with FK checks

---

## 13) CLI & API

- New in Python only
- Py: `cli/main.py`, `app/main.py`
- Commands: `generate`, `preview`, `import`
- API (optional): `POST /maps/generate`

---

## 14) Leaflet Previews

- New in Python only
- Template HTML with togglable layers
- Generated per stage and as `full_map.html`

---

## Execution Checklist per File

1. Translate JS → Python module
2. Write seeded parity tests
3. Produce GeoJSON artifacts
4. Generate Leaflet preview
5. Validate geometry + schema
6. Mark progress in `docs/porting-status.md`
