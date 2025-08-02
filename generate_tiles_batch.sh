#!/bin/bash

# Script pour générer toutes les tuiles 3D avec pg2b3dm
# Usage: ./generate_tiles_batch.sh [map_id]

set -e  # Exit on any error

# Configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-py_fmg}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-password}
TILES_DIR=${TILES_DIR:-./tiles}
MAP_ID=${1:-"latest"}

echo "🚀 Starting 3D tile generation for map: $MAP_ID"
echo "📊 Database: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
echo "📁 Output directory: $TILES_DIR"

# Create output directory
mkdir -p "$TILES_DIR"

# Function to run pg2b3dm with error handling
run_pg2b3dm() {
    local layer_name=$1
    local query=$2
    local output_dir=$3
    local geometric_error=${4:-2000}
    local max_features=${5:-500}
    
    echo "🔄 Generating $layer_name tiles..."
    
    # Create output directory for this layer
    mkdir -p "$output_dir"
    
    # Run pg2b3dm in Docker
    docker run --rm \
        --network host \
        -v "$PWD/$TILES_DIR:/tiles" \
        -e PGPASSWORD="$DB_PASSWORD" \
        geodan/pg2b3dm:latest \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "$query" \
        -t "${layer_name}_tiles" \
        --geometrycolumn "geom" \
        --attributescolumn "color" \
        -o "/tiles/$layer_name" \
        -g "$geometric_error" \
        --max_features_per_tile "$max_features"
    
    if [ $? -eq 0 ]; then
        echo "✅ $layer_name tiles generated successfully"
        # Check if tileset.json was created
        if [ -f "$TILES_DIR/$layer_name/tileset.json" ]; then
            echo "📄 Tileset JSON created: $TILES_DIR/$layer_name/tileset.json"
        else
            echo "⚠️  Warning: tileset.json not found for $layer_name"
        fi
    else
        echo "❌ Failed to generate $layer_name tiles"
        return 1
    fi
}

# Function to get map filter condition
get_map_filter() {
    if [ "$MAP_ID" = "latest" ]; then
        echo ""  # No filter for latest
    else
        echo "AND map_id = '$MAP_ID'"
    fi
}

# Function to get map filter condition
get_map_filter() {
    if [ "$MAP_ID" = "latest" ]; then
        # Get the latest map ID from the database
        LATEST_MAP_ID=$(docker exec py-fmg-db psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT id FROM maps ORDER BY created_at DESC LIMIT 1;" | tr -d ' ')
        if [ -n "$LATEST_MAP_ID" ]; then
            echo "AND map_id = '$LATEST_MAP_ID'"
        else
            echo ""  # No filter if no maps found
        fi
    else
        echo "AND map_id = '$MAP_ID'"
    fi
}

MAP_FILTER=$(get_map_filter)
echo "📍 Using map filter: $MAP_FILTER"

# 1. Generate terrain tiles
echo "🏔️  Generating terrain tiles..."
TERRAIN_QUERY="SELECT id, terrain_type, ST_Force3D(geom) as geom, color FROM terrain_3d WHERE geom IS NOT NULL $MAP_FILTER"
run_pg2b3dm "terrain" "$TERRAIN_QUERY" "$TILES_DIR/terrain" 5000 300

# 2. Generate settlement tiles
echo "🏘️  Generating settlement tiles..."
SETTLEMENTS_QUERY="SELECT id, name, settlement_class, pop_size, ST_Force3D(geom) as geom, color FROM settlements_3d WHERE geom IS NOT NULL $MAP_FILTER"
run_pg2b3dm "settlements" "$SETTLEMENTS_QUERY" "$TILES_DIR/settlements" 2000 200

# 3. Generate river tiles
echo "🌊 Generating river tiles..."
RIVERS_QUERY="SELECT id, name, river_class, discharge, ST_Force3D(geom) as geom, color FROM rivers_3d WHERE geom IS NOT NULL $MAP_FILTER"
run_pg2b3dm "rivers" "$RIVERS_QUERY" "$TILES_DIR/rivers" 1000 100

# 4. Generate culture tiles
echo "🎭 Generating culture tiles..."
CULTURES_QUERY="SELECT id, name, culture_type, ST_Force3D(geom) as geom, color FROM cultures_3d WHERE geom IS NOT NULL $MAP_FILTER"
run_pg2b3dm "cultures" "$CULTURES_QUERY" "$TILES_DIR/cultures" 3000 50

# 5. Generate state tiles
echo "🏛️  Generating state tiles..."
STATES_QUERY="SELECT id, name, boundary_type, ST_Force3D(geom) as geom, color FROM states_3d WHERE geom IS NOT NULL $MAP_FILTER"
run_pg2b3dm "states" "$STATES_QUERY" "$TILES_DIR/states" 4000 30

# Generate summary report
echo "📊 Generating tile summary..."
cat > "$TILES_DIR/generation_report.json" << EOF
{
    "generation_time": "$(date -Iseconds)",
    "map_id": "$MAP_ID",
    "database": {
        "host": "$DB_HOST",
        "port": "$DB_PORT",
        "name": "$DB_NAME"
    },
    "layers": {
        "terrain": {
            "directory": "terrain",
            "tileset": "$([ -f "$TILES_DIR/terrain/tileset.json" ] && echo "true" || echo "false")",
            "files": $(find "$TILES_DIR/terrain" -name "*.b3dm" 2>/dev/null | wc -l)
        },
        "settlements": {
            "directory": "settlements", 
            "tileset": "$([ -f "$TILES_DIR/settlements/tileset.json" ] && echo "true" || echo "false")",
            "files": $(find "$TILES_DIR/settlements" -name "*.b3dm" 2>/dev/null | wc -l)
        },
        "rivers": {
            "directory": "rivers",
            "tileset": "$([ -f "$TILES_DIR/rivers/tileset.json" ] && echo "true" || echo "false")",
            "files": $(find "$TILES_DIR/rivers" -name "*.b3dm" 2>/dev/null | wc -l)
        },
        "cultures": {
            "directory": "cultures",
            "tileset": "$([ -f "$TILES_DIR/cultures/tileset.json" ] && echo "true" || echo "false")",
            "files": $(find "$TILES_DIR/cultures" -name "*.b3dm" 2>/dev/null | wc -l)
        },
        "states": {
            "directory": "states",
            "tileset": "$([ -f "$TILES_DIR/states/tileset.json" ] && echo "true" || echo "false")",
            "files": $(find "$TILES_DIR/states" -name "*.b3dm" 2>/dev/null | wc -l)
        }
    },
    "total_files": $(find "$TILES_DIR" -name "*.b3dm" 2>/dev/null | wc -l),
    "total_size_mb": $(du -sm "$TILES_DIR" 2>/dev/null | cut -f1)
}
EOF

echo "📋 Generation complete! Summary:"
echo "   📁 Output directory: $TILES_DIR"
echo "   📄 Total .b3dm files: $(find "$TILES_DIR" -name "*.b3dm" 2>/dev/null | wc -l)"
echo "   💾 Total size: $(du -sh "$TILES_DIR" 2>/dev/null | cut -f1)"
echo "   📊 Report: $TILES_DIR/generation_report.json"

# Check if Cesium viewer is available
if [ -d "./viewer" ]; then
    echo "🌐 Cesium viewer available at: http://localhost:8081"
    echo "   Make sure to start the viewer service: docker-compose up cesium-viewer"
else
    echo "⚠️  Cesium viewer not found. Run: python generate_3d_tiles.py --create-viewer"
fi

echo "🎉 3D tile generation completed successfully!"

