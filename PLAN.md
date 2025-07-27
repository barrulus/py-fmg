# Python Port

This plan outlines the strategy for creating a headless map generation service by porting the logic from Fantasy Map Generator (FMG) into a Python application, using a powerful PostGIS backend designed for a procedural RPG, with clear pathways for future expansion to street-level detail.

---

## **Project Plan: FMG-to-RPG Geospatial Engine**

### **1. Executive Summary**

This document outlines a strategic plan to create a headless procedural map generation service. The core of this project involves porting the world-generation algorithms from Azgaar's Fantasy Map Generator (FMG) into a standalone Python application. This service will leverage Python's premier geospatial libraries to store the generated map data in a PostGIS-enabled PostgreSQL database.

The architecture is designed specifically for use by a procedural Role-Playing Game (RPG), enabling complex, real-time spatial queries about the game world. Furthermore, the plan establishes a hierarchical foundation that allows for future expansion into on-demand, detailed street-level map generation for cities and towns. The primary tool for visualization and authoring will be QGIS, connected directly to the PostGIS backend.

### **2. Core Philosophy & Guiding Principles**

*   **True Logic Port, Not UI Automation:** We will re-implement the core FMG algorithms in Python. This ensures a robust, efficient, and maintainable system, free from dependencies on the FMG web interface.
*   **Hierarchical Procedural Generation:** The system will operate on multiple levels of detail. FMG logic generates the macro-scale (continents, states). Specialized generators will handle the micro-scale (cities, streets), using the macro output as their input.
*   **Data-First, Not File-First:** The canonical source of truth for all map data will be the PostGIS database. Static files (GeoJSON, etc.) are treated as temporary exports, not the primary data store. This is crucial for the RPG's dynamic query needs.
*   **PostGIS as the Engine, QGIS as the Cockpit:** PostGIS is the high-performance database backend that stores and queries the data. QGIS is the desktop client used to connect to the database for visualization, styling, and manual editing of the game world.

### **3. System Architecture**

The system consists of three primary components that communicate linearly:

```
┌─────────────────┐      ┌─────────────────────────────────┐      ┌────────────────────┐
│   API / CLI     ├──────►   Python Generation Service     ├──────►   PostGIS Database │
│ (FastAPI/Click) │      │ (FMG Logic + City Generators)   │      │ (PostgreSQL)       │
└─────────────────┘      └─────────────────────────────────┘      └────────────────────┘
```

### **4. Database Schema Design**

A unified schema will store data from all maps and all levels of detail, linked via foreign keys. This enables powerful cross-layer queries.

#### **Level 1: Macro (FMG) Tables**

| Table: `maps` |                           |
| :------------ | :------------------------ |
| `map_id`      | `UUID` (Primary Key)      |
| `map_name`    | `TEXT`                    |
| `seed`        | `TEXT`                    |
| `bounds`      | `GEOMETRY(Polygon, 4326)` |
| `created_at`  | `TIMESTAMPTZ`             |

| Table: `states` |                                |
| :-------------- | :----------------------------- |
| `state_id`      | `UUID` (Primary Key)           |
| `map_id`        | `UUID` (Foreign Key to `maps`) |
| `geom`          | `GEOMETRY(Polygon, 4326)`      |
| `name`          | `TEXT`                         |
| ...             | *Other state properties*       |

| Table: `burgs` (Cities/Towns) |                                |
| :---------------------------- | :----------------------------- |
| `burg_id`                     | `UUID` (Primary Key)           |
| `map_id`                      | `UUID` (Foreign Key to `maps`) |
| `geom`                        | `GEOMETRY(Point, 4326)`        |
| `name`                        | `TEXT`                         |
| `population`                  | `INTEGER`                      |
| `is_port`                     | `BOOLEAN`                      |
| ...                           | *Other burg properties*        |

*(Similar tables for `rivers`, `biomes`, etc., all linked with `map_id`)*

---

#### **Level 2: Micro (Street-Level) Tables - For Future Expansion**

| Table: `districts` |                                 |
| :----------------- | :------------------------------ |
| `district_id`      | `UUID` (Primary Key)            |
| `burg_id`          | `UUID` (Foreign Key to `burgs`) |
| `geom`             | `GEOMETRY(Polygon, 4326)`       |
| `name`             | `TEXT` ("The Docks")            |
| `type`             | `TEXT` ("commercial")           |

| Table: `roads` |                                 |
| :------------- | :------------------------------ |
| `road_id`      | `UUID` (Primary Key)            |
| `burg_id`      | `UUID` (Foreign Key to `burgs`) |
| `geom`         | `GEOMETRY(LineString, 4326)`    |
| `class`        | `TEXT` ("main", "alley")        |
| `width`        | `REAL`                          |

| Table: `buildings` |                                 |
| :----------------- | :------------------------------ |
| `building_id`      | `UUID` (Primary Key)            |
| `burg_id`          | `UUID` (Foreign Key to `burgs`) |
| `geom`             | `GEOMETRY(Polygon, 4326)`       |
| `type`             | `TEXT` ("tavern", "house")      |
| `stories`          | `INTEGER`                       |

### **5. Implementation Plan**

#### **Phase 1: The Foundation - World-Scale Generation**

1.  **Port FMG's Core Logic:**
    *   Re-implement FMG's key JavaScript algorithms in a dedicated Python module.
    *   **Focus Areas:**
        *   **Graph Creation:** Use `scipy.spatial.Voronoi` to replicate FMG's base cell structure.
        *   **Heightmap Generation:** Implement Perlin/Simplex noise functions using the `noise` library.
        *   **Hydrology Simulation:** Port the water flow and river creation logic using graph traversal algorithms.
        *   **Attribute Assignment:** Re-create the logic for assigning biomes, cultures, and states to cells.

2.  **Develop the Headless Server:**
    *   Use **FastAPI** to build the REST API.
    *   Create the initial endpoint: `POST /maps/generate`.
    *   This endpoint will take a map name and generation parameters (seed, etc.).
    *   The service will run the ported FMG logic, generate a new `map_id`, and insert the map record into the `maps` table.

3.  **Implement the PostGIS Exporter:**
    *   Use **GeoPandas** and **SQLAlchemy** for GIS data handling and database communication.
    *   Convert the in-memory generated features (states, rivers, burgs) into GeoDataFrames.
    *   Add the correct `map_id` to every feature record.
    *   Append the GeoDataFrames to the appropriate PostGIS tables (`states`, `burgs`, etc.) using `gdf.to_postgis(..., if_exists='append')`.

#### **Phase 2: The Expansion - Street-Scale Generation**

1.  **Develop a Specialized City Generator Module:**
    *   Create a new Python class/module, `CityGenerator`, separate from the FMG logic.
    *   This module will take a `burg_id` and its properties (population, location, features) as input.
    *   **Implement Urban Generation Algorithms:**
        *   Agent-based systems or L-Systems for organic road network generation.
        *   Grid-based logic for planned city districts.
        *   Parcel subdivision algorithms to create lots for buildings.

2.  **Create the City Generation API Endpoint:**
    *   Add a new endpoint: `POST /burgs/{burg_id}/generate_city_detail`.
    *   This endpoint will trigger the `CityGenerator` for a specific city on-demand.
    *   The generator will produce districts, roads, and building footprints.
    *   The resulting GeoDataFrames will be appended to the `districts`, `roads`, and `buildings` tables in PostGIS, linked by the `burg_id`.

#### **Phase 3: Integration and Usage**

1.  **RPG Engine Integration:**
    *   The game server connects directly to the PostGIS database.
    *   It performs real-time spatial queries to inform game logic.
    *   **Example Queries:**
        *   *What state is the player in?* (`SELECT * FROM states WHERE ST_Contains(geom, player_location);`)
        *   *Are there any taverns within 50 meters?* (`SELECT * FROM buildings WHERE type = 'tavern' AND ST_DWithin(geom, player_location, 50);`)
        *   *Find the nearest road.* (`SELECT * FROM roads ORDER BY geom <-> player_location LIMIT 1;`)

2.  **QGIS Workflow:**
    *   Connect QGIS directly to the PostGIS database.
    *   Load tables (`states`, `burgs`, `roads`, `buildings`) as layers.
    *   Use **scale-dependent visibility** to automatically show/hide layers as you zoom, creating a seamless transition from world map to street view.
    *   Use **rule-based filtering** to display a single map at a time (e.g., `filter: "map_id" = '...'`).

### **6. Technology Stack Summary**

*   **Language:** Python 3.10+
*   **API Framework:** FastAPI
*   **Geospatial Processing:** GeoPandas, Shapely, Rasterio
*   **Database:** PostgreSQL 14+ with PostGIS 3+ extension
*   **Database Connector:** SQLAlchemy with Psycopg2
*   **Numerical/Scientific:** NumPy, SciPy
*   **Visualization/Authoring:** QGIS 3.x

### **7. Success Metrics**

*   **Performance:** World map generation (< 60 seconds). On-demand city generation (< 20 seconds). Median RPG spatial query response time (< 50ms).
*   **Data Integrity:** All generated geometries must be valid (no self-intersections). Foreign key relationships must be 100% consistent.
*   **Functionality:** The system must be able to answer the key RPG questions (location, proximity, pathing) via SQL queries.
*   **Scalability:** The database must handle dozens of generated world maps and thousands of detailed cities without significant performance degradation.
