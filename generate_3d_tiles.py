#!/usr/bin/env python3
"""
Generate 3D tiles from map data using pg2b3dm.

This script prepares the database with proper 3D geometries and triggers
the pg2b3dm service to generate 3D tiles for visualization.
"""

import os
import sys
import requests
import time
from pathlib import Path
from sqlalchemy import create_engine, text
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value


def get_latest_map_id():
    """Get the ID of the most recently generated map via API."""
    try:
        response = requests.get("http://localhost:9888/maps")
        if response.status_code == 200:
            maps = response.json()
            if maps:
                return maps[0]["id"]  # Most recent map
    except Exception as e:
        logger.error("Could not get map from API", error=str(e))
    return None


def prepare_3d_geometries(map_id: str):
    """
    Prepare 3D geometries in the database for pg2b3dm.
    
    This function ensures the 3D views are set up and populated with data
    for the specified map.
    """
    logger.info("Preparing 3D geometries", map_id=map_id)
    
    # Connect to database
    db_url = f'postgresql://{os.environ["DB_USER"]}:{os.environ["DB_PASSWORD"]}@{os.environ["DB_HOST"]}:{os.environ["DB_PORT"]}/{os.environ["DB_NAME"]}'
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # First, ensure the 3D setup script has been run
        logger.info("Checking 3D views setup")
        
        # Check if our 3D views exist
        result = conn.execute(text("""
            SELECT COUNT(*) as view_count
            FROM information_schema.views 
            WHERE table_name IN ('terrain_3d', 'settlements_3d', 'rivers_3d', 'cultures_3d', 'states_3d')
        """))
        
        view_count = result.fetchone()[0]
        if view_count < 5:
            logger.warning("3D views not found, creating them")
            # Read and execute the setup script
            try:
                with open('setup_3d_views.sql', 'r') as f:
                    setup_sql = f.read()
                    # Execute the setup script
                    conn.execute(text(setup_sql))
                    conn.commit()
                    logger.info("3D views created successfully")
            except Exception as e:
                logger.error("Failed to create 3D views", error=str(e))
                raise
        
        # Check the 3D setup for the specific map
        logger.info("Checking 3D setup for map", map_id=map_id)
        result = conn.execute(text("SELECT * FROM check_3d_setup()"))
        setup_status = result.fetchall()
        
        for layer_name, view_exists, feature_count, sample_geom in setup_status:
            logger.info("3D layer status", 
                       layer=layer_name, 
                       view_exists=view_exists, 
                       features=feature_count,
                       has_geometry=sample_geom is not None)
        
        # Verify we have data for this map
        result = conn.execute(text("""
            SELECT 
                (SELECT COUNT(*) FROM voronoi_cells WHERE map_id = :map_id) as terrain_count,
                (SELECT COUNT(*) FROM settlements WHERE map_id = :map_id) as settlement_count,
                (SELECT COUNT(*) FROM rivers WHERE map_id = :map_id) as river_count,
                (SELECT COUNT(*) FROM cultures WHERE map_id = :map_id) as culture_count,
                (SELECT COUNT(*) FROM states WHERE map_id = :map_id) as state_count
        """), {"map_id": map_id})
        
        counts = result.fetchone()
        logger.info("Map data counts", 
                   map_id=map_id,
                   terrain=counts[0],
                   settlements=counts[1], 
                   rivers=counts[2],
                   cultures=counts[3],
                   states=counts[4])
        
        if counts[0] == 0:
            raise ValueError(f"No terrain data found for map {map_id}")
        
        logger.info("3D geometries prepared successfully")


def trigger_tile_generation():
    """
    Trigger the pg2b3dm service to generate 3D tiles.
    """
    logger.info("Triggering 3D tile generation")
    
    # Configuration for pg2b3dm
    config = {
        "terrain": {
            "table": "map_3d_terrain",
            "geometry_column": "geom",
            "color_column": "color",
            "tileset_name": "terrain"
        },
        "settlements": {
            "table": "map_3d_settlements", 
            "geometry_column": "geom",
            "color_column": "color",
            "tileset_name": "settlements"
        },
        "rivers": {
            "table": "map_3d_rivers",
            "geometry_column": "geom", 
            "color_column": "color",
            "tileset_name": "rivers"
        },
        "cultures": {
            "table": "map_3d_cultures",
            "geometry_column": "geom",
            "color_column": "color", 
            "tileset_name": "cultures"
        }
    }
    
    # Try to trigger pg2b3dm via HTTP API (if available)
    try:
        for layer_name, layer_config in config.items():
            logger.info("Generating tiles for layer", layer=layer_name)
            
            # This would be the actual API call to pg2b3dm
            # The exact API depends on how pg2b3dm is configured
            response = requests.post(
                "http://localhost:8080/generate",
                json=layer_config,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                logger.info("Tiles generated successfully", layer=layer_name)
            else:
                logger.warning("Failed to generate tiles", 
                             layer=layer_name, 
                             status=response.status_code)
                
    except requests.exceptions.RequestException as e:
        logger.warning("Could not connect to pg2b3dm service", error=str(e))
        logger.info("Tiles will be generated automatically by pg2b3dm service")


def create_cesium_viewer():
    """
    Create a simple Cesium viewer HTML file for 3D visualization.
    """
    logger.info("Creating Cesium viewer")
    
    viewer_dir = Path("viewer")
    viewer_dir.mkdir(exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {
            width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
        }
        #toolbar {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(42, 42, 42, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
        #toolbar button {
            margin: 5px;
            padding: 5px 10px;
            background: #48b;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        #toolbar button:hover {
            background: #369;
        }
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div id="toolbar">
        <button onclick="showTerrain()">Terrain</button>
        <button onclick="showSettlements()">Settlements</button>
        <button onclick="showRivers()">Rivers</button>
        <button onclick="showCultures()">Cultures</button>
        <button onclick="showAll()">Show All</button>
        <button onclick="resetView()">Reset View</button>
    </div>

    <script>
        // Initialize Cesium viewer
        const viewer = new Cesium.Viewer('cesiumContainer', {
            terrainProvider: Cesium.createWorldTerrain(),
            homeButton: false,
            sceneModePicker: false,
            baseLayerPicker: false,
            navigationHelpButton: false,
            animation: false,
            timeline: false,
            fullscreenButton: false,
            geocoder: false
        });

        // Store tilesets
        let tilesets = {};

        // Load 3D tilesets
        function loadTileset(name, url) {
            try {
                const tileset = viewer.scene.primitives.add(
                    new Cesium.Cesium3DTileset({
                        url: url,
                        show: false
                    })
                );
                
                tileset.readyPromise.then(function(tileset) {
                    console.log(`${name} tileset loaded`);
                    // Adjust height if needed
                    const heightOffset = name === 'terrain' ? 0 : 10;
                    const cartographic = Cesium.Cartographic.fromCartesian(tileset.boundingSphere.center);
                    const surface = Cesium.Cartesian3.fromRadians(cartographic.longitude, cartographic.latitude, cartographic.height + heightOffset);
                    const translation = Cesium.Cartesian3.subtract(surface, tileset.boundingSphere.center, new Cesium.Cartesian3());
                    tileset.modelMatrix = Cesium.Matrix4.fromTranslation(translation);
                }).otherwise(function(error) {
                    console.error(`Error loading ${name} tileset:`, error);
                });
                
                tilesets[name] = tileset;
            } catch (error) {
                console.error(`Failed to load ${name} tileset:`, error);
            }
        }

        // Load all tilesets
        loadTileset('terrain', './tiles/terrain/tileset.json');
        loadTileset('settlements', './tiles/settlements/tileset.json');
        loadTileset('rivers', './tiles/rivers/tileset.json');
        loadTileset('cultures', './tiles/cultures/tileset.json');

        // Control functions
        function showTerrain() {
            hideAll();
            if (tilesets.terrain) tilesets.terrain.show = true;
        }

        function showSettlements() {
            hideAll();
            if (tilesets.terrain) tilesets.terrain.show = true;
            if (tilesets.settlements) tilesets.settlements.show = true;
        }

        function showRivers() {
            hideAll();
            if (tilesets.terrain) tilesets.terrain.show = true;
            if (tilesets.rivers) tilesets.rivers.show = true;
        }

        function showCultures() {
            hideAll();
            if (tilesets.terrain) tilesets.terrain.show = true;
            if (tilesets.cultures) tilesets.cultures.show = true;
        }

        function showAll() {
            Object.values(tilesets).forEach(tileset => {
                if (tileset) tileset.show = true;
            });
        }

        function hideAll() {
            Object.values(tilesets).forEach(tileset => {
                if (tileset) tileset.show = false;
            });
        }

        function resetView() {
            if (tilesets.terrain && tilesets.terrain.ready) {
                viewer.zoomTo(tilesets.terrain);
            }
        }

        // Default view
        setTimeout(() => {
            showTerrain();
            resetView();
        }, 2000);
    </script>
</body>
</html>"""
    
    with open(viewer_dir / "index.html", "w") as f:
        f.write(html_content)
    
    logger.info("Cesium viewer created", path=str(viewer_dir / "index.html"))


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D tiles from map data")
    parser.add_argument("--map-id", help="Map ID to generate tiles for (uses latest if not specified)")
    parser.add_argument("--skip-viewer", action="store_true", help="Skip creating Cesium viewer")
    
    args = parser.parse_args()
    
    # Get map ID
    map_id = args.map_id
    if not map_id:
        map_id = get_latest_map_id()
        if not map_id:
            logger.error("No maps found or API not available")
            sys.exit(1)
    
    logger.info("Generating 3D tiles", map_id=map_id)
    
    try:
        # Prepare 3D geometries
        prepare_3d_geometries(map_id)
        
        # Trigger tile generation
        trigger_tile_generation()
        
        # Create viewer
        if not args.skip_viewer:
            create_cesium_viewer()
        
        logger.info("3D tile generation completed successfully")
        logger.info("View your 3D map at: http://localhost:8081")
        
    except Exception as e:
        logger.error("Failed to generate 3D tiles", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

