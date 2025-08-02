-- Setup 3D views for pg2b3dm tile generation
-- This script creates the necessary database views and functions for 3D visualization

-- Enable PostGIS 3D functions if not already enabled
CREATE EXTENSION IF NOT EXISTS postgis;

-- Function to safely extrude geometries to 3D
CREATE OR REPLACE FUNCTION safe_extrude(geom geometry, dx float, dy float, dz float)
RETURNS geometry AS $$
BEGIN
    -- Check if geometry is valid and not null
    IF geom IS NULL OR NOT ST_IsValid(geom) THEN
        RETURN NULL;
    END IF;
    
    -- Force to 2D first, then extrude
    RETURN ST_Extrude(ST_Force2D(geom), dx, dy, dz);
EXCEPTION
    WHEN OTHERS THEN
        -- Return NULL if extrusion fails
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create 3D terrain view for pg2b3dm (filtered by map_id)
DROP VIEW IF EXISTS terrain_3d CASCADE;
CREATE VIEW terrain_3d AS
SELECT 
    cell_index as id,
    map_id,
    height,
    is_land,
    biome,
    safe_extrude(
        geometry, 
        0, 0, 
        GREATEST(COALESCE(height, 0) * 5, 1)  -- Scale height and ensure minimum 1 unit
    ) as geom,
    CASE 
        WHEN NOT COALESCE(is_land, false) THEN '#4169E1'  -- Blue for water
        WHEN COALESCE(height, 0) > 100 THEN '#8B4513'     -- Brown for mountains
        WHEN COALESCE(height, 0) > 50 THEN '#228B22'      -- Green for hills
        WHEN COALESCE(biome, '') LIKE '%Desert%' THEN '#F4A460'  -- Sandy brown for desert
        WHEN COALESCE(biome, '') LIKE '%Forest%' THEN '#006400'  -- Dark green for forest
        WHEN COALESCE(biome, '') LIKE '%Grassland%' THEN '#90EE90'  -- Light green for grassland
        ELSE '#90EE90'  -- Default light green for land
    END as color,
    -- Additional attributes for styling
    CASE 
        WHEN NOT COALESCE(is_land, false) THEN 'water'
        WHEN COALESCE(height, 0) > 100 THEN 'mountain'
        WHEN COALESCE(height, 0) > 50 THEN 'hill'
        ELSE 'lowland'
    END as terrain_type
FROM voronoi_cells 
WHERE geometry IS NOT NULL
AND ST_IsValid(geometry);

-- Create 3D settlements view for pg2b3dm (filtered by map_id)
DROP VIEW IF EXISTS settlements_3d CASCADE;
CREATE VIEW settlements_3d AS
SELECT 
    settlement_index as id,
    map_id,
    name,
    settlement_type,
    population,
    is_capital,
    safe_extrude(
        ST_Buffer(
            geometry, 
            CASE 
                WHEN COALESCE(is_capital, false) THEN 500
                WHEN COALESCE(settlement_type, '') = 'city' THEN 300
                WHEN COALESCE(settlement_type, '') = 'town' THEN 200
                ELSE 100
            END
        ), 
        0, 0, 
        CASE 
            WHEN COALESCE(is_capital, false) THEN 100
            WHEN COALESCE(settlement_type, '') = 'city' THEN 60
            WHEN COALESCE(settlement_type, '') = 'town' THEN 40
            ELSE 20
        END
    ) as geom,
    CASE 
        WHEN COALESCE(is_capital, false) THEN '#FF0000'  -- Red for capitals
        WHEN COALESCE(settlement_type, '') = 'city' THEN '#FFA500'  -- Orange for cities
        WHEN COALESCE(settlement_type, '') = 'town' THEN '#FFD700'  -- Gold for towns
        ELSE '#8B4513'  -- Brown for villages
    END as color,
    -- Additional attributes
    COALESCE(population, 0) as pop_size,
    CASE 
        WHEN COALESCE(is_capital, false) THEN 'capital'
        WHEN COALESCE(settlement_type, '') = 'city' THEN 'city'
        WHEN COALESCE(settlement_type, '') = 'town' THEN 'town'
        ELSE 'village'
    END as settlement_class
FROM settlements 
WHERE geometry IS NOT NULL
AND ST_IsValid(geometry);

-- Create 3D rivers view for pg2b3dm (filtered by map_id)
DROP VIEW IF EXISTS rivers_3d CASCADE;
CREATE VIEW rivers_3d AS
SELECT 
    river_index as id,
    map_id,
    name,
    length_km,
    discharge_m3s,
    safe_extrude(
        ST_Buffer(
            geometry, 
            GREATEST(COALESCE(discharge_m3s, 10) / 50, 5)  -- Width based on discharge, minimum 5 units
        ), 
        0, 0, 3  -- Rivers are slightly elevated for visibility
    ) as geom,
    '#0066CC' as color,  -- Blue for rivers
    -- Additional attributes
    COALESCE(discharge_m3s, 0) as discharge,
    CASE 
        WHEN COALESCE(discharge_m3s, 0) > 1000 THEN 'major'
        WHEN COALESCE(discharge_m3s, 0) > 100 THEN 'medium'
        ELSE 'minor'
    END as river_class
FROM rivers 
WHERE geometry IS NOT NULL
AND ST_IsValid(geometry)
AND COALESCE(discharge_m3s, 0) > 5;  -- Only show rivers with significant discharge

-- Create 3D cultures view for pg2b3dm (filtered by map_id)
DROP VIEW IF EXISTS cultures_3d CASCADE;
CREATE VIEW cultures_3d AS
SELECT 
    culture_index as id,
    map_id,
    name,
    type,
    safe_extrude(
        ST_Boundary(geometry), 
        0, 0, 25  -- Cultural boundary walls
    ) as geom,
    COALESCE(color, '#888888') as color,
    -- Additional attributes
    COALESCE(type, 'Generic') as culture_type
FROM cultures 
WHERE geometry IS NOT NULL
AND ST_IsValid(geometry)
AND ST_GeometryType(geometry) IN ('ST_Polygon', 'ST_MultiPolygon');

-- Create 3D states view for pg2b3dm (filtered by map_id)
DROP VIEW IF EXISTS states_3d CASCADE;
CREATE VIEW states_3d AS
SELECT 
    state_index as id,
    map_id,
    name,
    safe_extrude(
        ST_Boundary(geometry), 
        0, 0, 50  -- Political boundary walls, higher than cultural boundaries
    ) as geom,
    COALESCE(color, '#666666') as color,
    -- Additional attributes
    'political' as boundary_type
FROM states 
WHERE geometry IS NOT NULL
AND ST_IsValid(geometry)
AND ST_GeometryType(geometry) IN ('ST_Polygon', 'ST_MultiPolygon');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_terrain_3d_geom ON voronoi_cells USING GIST (geometry) WHERE geometry IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_settlements_3d_geom ON settlements USING GIST (geometry) WHERE geometry IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rivers_3d_geom ON rivers USING GIST (geometry) WHERE geometry IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cultures_3d_geom ON cultures USING GIST (geometry) WHERE geometry IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_states_3d_geom ON states USING GIST (geometry) WHERE geometry IS NOT NULL;

-- Grant permissions for pg2b3dm user (adjust username as needed)
GRANT SELECT ON terrain_3d TO PUBLIC;
GRANT SELECT ON settlements_3d TO PUBLIC;
GRANT SELECT ON rivers_3d TO PUBLIC;
GRANT SELECT ON cultures_3d TO PUBLIC;
GRANT SELECT ON states_3d TO PUBLIC;

-- Create a summary view for debugging
DROP VIEW IF EXISTS tileset_summary CASCADE;
CREATE VIEW tileset_summary AS
SELECT 
    'terrain' as layer,
    COUNT(*) as feature_count,
    ST_Extent(geom) as bbox
FROM terrain_3d
WHERE geom IS NOT NULL
UNION ALL
SELECT 
    'settlements' as layer,
    COUNT(*) as feature_count,
    ST_Extent(geom) as bbox
FROM settlements_3d
WHERE geom IS NOT NULL
UNION ALL
SELECT 
    'rivers' as layer,
    COUNT(*) as feature_count,
    ST_Extent(geom) as bbox
FROM rivers_3d
WHERE geom IS NOT NULL
UNION ALL
SELECT 
    'cultures' as layer,
    COUNT(*) as feature_count,
    ST_Extent(geom) as bbox
FROM cultures_3d
WHERE geom IS NOT NULL
UNION ALL
SELECT 
    'states' as layer,
    COUNT(*) as feature_count,
    ST_Extent(geom) as bbox
FROM states_3d
WHERE geom IS NOT NULL;

-- Function to check 3D setup
CREATE OR REPLACE FUNCTION check_3d_setup()
RETURNS TABLE(
    layer_name text,
    view_exists boolean,
    feature_count bigint,
    sample_geometry text
) AS $$
BEGIN
    -- Check terrain
    RETURN QUERY
    SELECT 
        'terrain'::text,
        EXISTS(SELECT 1 FROM information_schema.views WHERE table_name = 'terrain_3d'),
        (SELECT COUNT(*) FROM terrain_3d WHERE geom IS NOT NULL),
        (SELECT ST_AsText(geom) FROM terrain_3d WHERE geom IS NOT NULL LIMIT 1);
    
    -- Check settlements
    RETURN QUERY
    SELECT 
        'settlements'::text,
        EXISTS(SELECT 1 FROM information_schema.views WHERE table_name = 'settlements_3d'),
        (SELECT COUNT(*) FROM settlements_3d WHERE geom IS NOT NULL),
        (SELECT ST_AsText(geom) FROM settlements_3d WHERE geom IS NOT NULL LIMIT 1);
    
    -- Check rivers
    RETURN QUERY
    SELECT 
        'rivers'::text,
        EXISTS(SELECT 1 FROM information_schema.views WHERE table_name = 'rivers_3d'),
        (SELECT COUNT(*) FROM rivers_3d WHERE geom IS NOT NULL),
        (SELECT ST_AsText(geom) FROM rivers_3d WHERE geom IS NOT NULL LIMIT 1);
    
    -- Check cultures
    RETURN QUERY
    SELECT 
        'cultures'::text,
        EXISTS(SELECT 1 FROM information_schema.views WHERE table_name = 'cultures_3d'),
        (SELECT COUNT(*) FROM cultures_3d WHERE geom IS NOT NULL),
        (SELECT ST_AsText(geom) FROM cultures_3d WHERE geom IS NOT NULL LIMIT 1);
    
    -- Check states
    RETURN QUERY
    SELECT 
        'states'::text,
        EXISTS(SELECT 1 FROM information_schema.views WHERE table_name = 'states_3d'),
        (SELECT COUNT(*) FROM states_3d WHERE geom IS NOT NULL),
        (SELECT ST_AsText(geom) FROM states_3d WHERE geom IS NOT NULL LIMIT 1);
END;
$$ LANGUAGE plpgsql;

