"""
3D Visualization API endpoints.

This module provides endpoints for generating and managing 3D visualizations
of maps, including tile generation and viewer integration.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
import structlog
import subprocess
import os
import json
from pathlib import Path
import asyncio

from ..db.connection import db
from ..db.models import Map

logger = structlog.get_logger()

# Create router for visualization endpoints
router = APIRouter(prefix="/maps/{map_id}/visualize", tags=["3D Visualization"])


# Pydantic models for visualization operations
class TileGenerationRequest(BaseModel):
    """Model for 3D tile generation requests."""
    
    layers: List[str] = Field(
        default=["terrain", "settlements", "rivers", "cultures", "states"],
        description="List of layers to generate tiles for"
    )
    force_regenerate: bool = Field(
        default=False, 
        description="Force regeneration even if tiles already exist"
    )
    quality: str = Field(
        default="medium",
        description="Tile quality: low, medium, high"
    )
    output_format: str = Field(
        default="3dtiles",
        description="Output format: 3dtiles, gltf"
    )


class TileGenerationResponse(BaseModel):
    """Response model for tile generation."""
    
    success: bool = Field(description="Whether generation succeeded")
    message: str = Field(description="Result message")
    job_id: str = Field(description="Background job ID")
    layers_generated: List[str] = Field(description="List of successfully generated layers")
    estimated_completion_time: Optional[int] = Field(
        default=None, 
        description="Estimated completion time in seconds"
    )
    tiles_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Information about generated tiles"
    )


class TileStatus(BaseModel):
    """Model for tile generation status."""
    
    layer: str = Field(description="Layer name")
    status: str = Field(description="Status: pending, generating, completed, failed")
    progress: float = Field(description="Progress percentage (0-100)")
    file_count: int = Field(description="Number of tile files generated")
    size_mb: float = Field(description="Total size in MB")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class VisualizationInfo(BaseModel):
    """Model for visualization information."""
    
    map_id: str = Field(description="Map ID")
    map_name: str = Field(description="Map name")
    tiles_available: bool = Field(description="Whether 3D tiles are available")
    viewer_url: str = Field(description="URL to 3D viewer")
    layers: List[TileStatus] = Field(description="Status of each layer")
    last_generated: Optional[str] = Field(default=None, description="Last generation timestamp")
    total_size_mb: float = Field(description="Total size of all tiles")


class ViewerConfig(BaseModel):
    """Model for viewer configuration."""
    
    initial_view: Dict[str, float] = Field(
        default={"longitude": 0, "latitude": 0, "height": 10000},
        description="Initial camera position"
    )
    enabled_layers: List[str] = Field(
        default=["terrain"],
        description="Initially enabled layers"
    )
    terrain_exaggeration: float = Field(
        default=1.0,
        description="Terrain height exaggeration factor"
    )
    lighting: Dict[str, Any] = Field(
        default={"enable_shadows": True, "ambient_light": 0.3},
        description="Lighting configuration"
    )


# Helper functions
def get_map_or_404(map_id: str):
    """Get map by ID or raise 404."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()
        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")
        return map_obj


def get_tiles_directory(map_id: str) -> Path:
    """Get the tiles directory for a specific map."""
    tiles_dir = Path("tiles") / map_id
    tiles_dir.mkdir(parents=True, exist_ok=True)
    return tiles_dir


def get_quality_settings(quality: str) -> Dict[str, Any]:
    """Get quality settings for tile generation."""
    settings = {
        "low": {
            "geometric_error": 8000,
            "max_features_per_tile": 200,
            "terrain_scale": 3
        },
        "medium": {
            "geometric_error": 4000,
            "max_features_per_tile": 500,
            "terrain_scale": 5
        },
        "high": {
            "geometric_error": 2000,
            "max_features_per_tile": 1000,
            "terrain_scale": 8
        }
    }
    return settings.get(quality, settings["medium"])


async def generate_tiles_async(map_id: str, layers: List[str], quality: str) -> Dict[str, Any]:
    """Generate 3D tiles asynchronously."""
    logger.info("Starting async tile generation", map_id=map_id, layers=layers)
    
    tiles_dir = get_tiles_directory(map_id)
    quality_settings = get_quality_settings(quality)
    
    results = {
        "layers_generated": [],
        "layers_failed": [],
        "total_files": 0,
        "total_size_mb": 0.0
    }
    
    # Generate tiles for each layer
    for layer in layers:
        try:
            logger.info("Generating tiles for layer", layer=layer, map_id=map_id)
            
            # Run the tile generation script
            cmd = [
                "./generate_tiles_batch.sh",
                map_id,
                layer,
                str(quality_settings["geometric_error"]),
                str(quality_settings["max_features_per_tile"])
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                results["layers_generated"].append(layer)
                logger.info("Layer tiles generated successfully", layer=layer)
                
                # Count files and calculate size
                layer_dir = tiles_dir / layer
                if layer_dir.exists():
                    files = list(layer_dir.glob("*.b3dm"))
                    results["total_files"] += len(files)
                    
                    size_bytes = sum(f.stat().st_size for f in files)
                    size_mb = size_bytes / (1024 * 1024)
                    results["total_size_mb"] += size_mb
                    
            else:
                results["layers_failed"].append({
                    "layer": layer,
                    "error": stderr.decode() if stderr else "Unknown error"
                })
                logger.error("Layer tile generation failed", 
                           layer=layer, 
                           error=stderr.decode() if stderr else "Unknown error")
                
        except Exception as e:
            results["layers_failed"].append({
                "layer": layer,
                "error": str(e)
            })
            logger.error("Exception during tile generation", layer=layer, error=str(e))
    
    return results


def get_tile_status(map_id: str) -> List[TileStatus]:
    """Get the status of tiles for each layer."""
    tiles_dir = get_tiles_directory(map_id)
    layers = ["terrain", "settlements", "rivers", "cultures", "states"]
    statuses = []
    
    for layer in layers:
        layer_dir = tiles_dir / layer
        
        if not layer_dir.exists():
            status = TileStatus(
                layer=layer,
                status="pending",
                progress=0.0,
                file_count=0,
                size_mb=0.0
            )
        else:
            # Check if tileset.json exists
            tileset_file = layer_dir / "tileset.json"
            if tileset_file.exists():
                # Count .b3dm files
                b3dm_files = list(layer_dir.glob("*.b3dm"))
                file_count = len(b3dm_files)
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in b3dm_files)
                size_mb = total_size / (1024 * 1024)
                
                status = TileStatus(
                    layer=layer,
                    status="completed",
                    progress=100.0,
                    file_count=file_count,
                    size_mb=size_mb
                )
            else:
                status = TileStatus(
                    layer=layer,
                    status="failed",
                    progress=0.0,
                    file_count=0,
                    size_mb=0.0,
                    error_message="Tileset file not found"
                )
        
        statuses.append(status)
    
    return statuses


# API Endpoints
@router.get("/info", response_model=VisualizationInfo)
async def get_visualization_info(map_id: str):
    """
    Get information about 3D visualization for a map.
    
    Returns the current status of 3D tiles, viewer URL, and layer information.
    """
    logger.info("Getting visualization info", map_id=map_id)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Get tile status
    tile_statuses = get_tile_status(map_id)
    
    # Check if any tiles are available
    tiles_available = any(status.status == "completed" for status in tile_statuses)
    
    # Calculate total size
    total_size_mb = sum(status.size_mb for status in tile_statuses)
    
    # Get last generation time (from tileset files)
    tiles_dir = get_tiles_directory(map_id)
    last_generated = None
    for layer_dir in tiles_dir.iterdir():
        if layer_dir.is_dir():
            tileset_file = layer_dir / "tileset.json"
            if tileset_file.exists():
                mtime = tileset_file.stat().st_mtime
                if last_generated is None or mtime > last_generated:
                    last_generated = mtime
    
    if last_generated:
        from datetime import datetime
        last_generated = datetime.fromtimestamp(last_generated).isoformat()
    
    return VisualizationInfo(
        map_id=map_id,
        map_name=map_obj.name,
        tiles_available=tiles_available,
        viewer_url=f"http://localhost:8081?map={map_id}",
        layers=tile_statuses,
        last_generated=last_generated,
        total_size_mb=total_size_mb
    )


@router.post("/generate", response_model=TileGenerationResponse)
async def generate_3d_tiles(
    map_id: str, 
    request: TileGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate 3D tiles for the specified map.
    
    This endpoint starts the tile generation process in the background
    and returns immediately with a job ID for tracking progress.
    """
    logger.info("3D tile generation requested", map_id=map_id, layers=request.layers)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Check if tiles already exist and force_regenerate is False
    if not request.force_regenerate:
        tile_statuses = get_tile_status(map_id)
        existing_layers = [
            status.layer for status in tile_statuses 
            if status.status == "completed" and status.layer in request.layers
        ]
        
        if existing_layers:
            logger.info("Tiles already exist", map_id=map_id, existing_layers=existing_layers)
            if len(existing_layers) == len(request.layers):
                return TileGenerationResponse(
                    success=True,
                    message="Tiles already exist. Use force_regenerate=true to regenerate.",
                    job_id="existing",
                    layers_generated=existing_layers,
                    estimated_completion_time=0
                )
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Estimate completion time based on layers and quality
    layer_times = {
        "terrain": 180,  # 3 minutes
        "settlements": 60,  # 1 minute
        "rivers": 90,   # 1.5 minutes
        "cultures": 60, # 1 minute
        "states": 60    # 1 minute
    }
    
    quality_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0}
    estimated_time = sum(
        layer_times.get(layer, 60) for layer in request.layers
    ) * quality_multiplier.get(request.quality, 1.0)
    
    # Start background task
    background_tasks.add_task(
        generate_tiles_async,
        map_id,
        request.layers,
        request.quality
    )
    
    return TileGenerationResponse(
        success=True,
        message=f"3D tile generation started for {len(request.layers)} layers",
        job_id=job_id,
        layers_generated=[],
        estimated_completion_time=int(estimated_time)
    )


@router.get("/status", response_model=List[TileStatus])
async def get_tile_generation_status(map_id: str):
    """
    Get the current status of 3D tile generation for all layers.
    
    Returns detailed information about each layer's generation status,
    including progress, file counts, and any errors.
    """
    logger.info("Getting tile generation status", map_id=map_id)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    return get_tile_status(map_id)


@router.delete("/tiles")
async def delete_tiles(map_id: str, layers: Optional[List[str]] = None):
    """
    Delete 3D tiles for the specified map and layers.
    
    If no layers are specified, deletes all tiles for the map.
    """
    logger.info("Deleting tiles", map_id=map_id, layers=layers)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    tiles_dir = get_tiles_directory(map_id)
    deleted_layers = []
    
    if layers is None:
        # Delete all tiles
        if tiles_dir.exists():
            import shutil
            shutil.rmtree(tiles_dir)
            deleted_layers = ["all"]
    else:
        # Delete specific layers
        for layer in layers:
            layer_dir = tiles_dir / layer
            if layer_dir.exists():
                import shutil
                shutil.rmtree(layer_dir)
                deleted_layers.append(layer)
    
    return {
        "success": True,
        "message": f"Deleted tiles for layers: {', '.join(deleted_layers)}",
        "deleted_layers": deleted_layers
    }


@router.get("/viewer")
async def get_viewer_url(map_id: str, config: Optional[ViewerConfig] = None):
    """
    Get the 3D viewer URL with optional configuration.
    
    Returns a URL to the Cesium-based 3D viewer with the specified
    configuration parameters.
    """
    logger.info("Getting viewer URL", map_id=map_id)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    # Check if tiles are available
    tile_statuses = get_tile_status(map_id)
    available_layers = [
        status.layer for status in tile_statuses 
        if status.status == "completed"
    ]
    
    if not available_layers:
        raise HTTPException(
            status_code=404, 
            detail="No 3D tiles available for this map. Generate tiles first."
        )
    
    # Build viewer URL with parameters
    base_url = f"http://localhost:8081"
    params = [f"map={map_id}"]
    
    if config:
        if config.enabled_layers:
            # Only include layers that are actually available
            valid_layers = [
                layer for layer in config.enabled_layers 
                if layer in available_layers
            ]
            if valid_layers:
                params.append(f"layers={','.join(valid_layers)}")
        
        if config.terrain_exaggeration != 1.0:
            params.append(f"exaggeration={config.terrain_exaggeration}")
    
    viewer_url = f"{base_url}?{'&'.join(params)}"
    
    return {
        "viewer_url": viewer_url,
        "available_layers": available_layers,
        "map_id": map_id
    }


@router.post("/viewer/config")
async def update_viewer_config(map_id: str, config: ViewerConfig):
    """
    Update the viewer configuration for a map.
    
    Saves the configuration to be used as default when opening the viewer.
    """
    logger.info("Updating viewer config", map_id=map_id)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    # Save config to file
    tiles_dir = get_tiles_directory(map_id)
    config_file = tiles_dir / "viewer_config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config.dict(), f, indent=2)
    
    return {
        "success": True,
        "message": "Viewer configuration updated",
        "config": config.dict()
    }

