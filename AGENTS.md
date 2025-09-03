# AGENTS.md — Codex-Driven Port for **py-fmg**

This document defines the **agents** and their responsibilities in the Codex-led port of **Azgaar's Fantasy Map Generator (FMG)** from JavaScript into a **headless Python + PostGIS** environment named **py-fmg**.

> **Key updates**
>
> - Project name: **py-fmg**
> - Primary data interchange is **GeoJSON** produced by py-fmg (no QGIS dependency).
> - We will ingest GeoJSON directly into PostGIS.
> - For previews and debugging, we will generate **Leaflet-based HTML** viewers and optional static PNGs.
> - GeoJSON datasets to support: **cells, burgs, states, provinces, routes, rivers, regiments, markers**.

---

## Agents

### 1) Codex — Orchestrator

- Coordinates all tasks.
- Ensures 1:1 logic parity with FMG.
- Produces validated GeoJSON and Leaflet previews.

### 2) Translator Agent

- Converts each FMG JavaScript file into a Python module.
- Keeps logic identical, substitutes with Python libraries.

### 3) Generation Agent

- Runs procedural steps: mesh, heightmap, hydrology, biomes, burgs, states, routes, regiments, markers.
- Produces in-memory feature sets and GeoJSON.

### 4) Persistence Agent

- Imports GeoJSON into PostGIS.
- Enforces geometry validity, SRID, and foreign keys.

### 5) Test & Validation Agent

- Seeded tests comparing snapshots to FMG.
- GeoJSON schema and geometry validation.
- Leaflet previews and optional PNGs.

### 6) API Agent

- CLI and optional FastAPI endpoints to generate maps and expose artifacts.

### 7) Visualization Agent (Leaflet)

- Generates HTML previews with layer toggles.
- Styles by type (states categorical, rivers weighted, burgs/markers icons).

### 8) Success Metrics

- Seed-locked determinism.
- Valid geometries.
- GeoJSON importable to PostGIS without repair.
- Generation < 60s.
