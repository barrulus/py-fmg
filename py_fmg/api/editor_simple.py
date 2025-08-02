"""
Simplified map editing API endpoints.

This module provides basic editing capabilities for maps.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from sqlalchemy import text

from ..db.connection import db

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/maps/{map_id}/edit", tags=["editing"])


# Pydantic models
class EditResponse(BaseModel):
    """Standard response for editing operations."""
    
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable message")
    affected_count: int = Field(default=0, description="Number of items affected")
    regenerate_required: bool = Field(default=False, description="Whether regeneration is needed")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class TerrainEdit(BaseModel):
    """Model for terrain editing requests."""
    
    cell_indices: List[int] = Field(description="List of cell indices to modify")
    operation: str = Field(description="Operation: set_height, adjust_height, set_biome")
    value: Optional[float] = Field(default=None, description="Value for height operations")
    biome: Optional[str] = Field(default=None, description="Biome name for biome operations")


class SettlementEdit(BaseModel):
    """Model for settlement editing requests."""
    
    settlement_index: int = Field(description="Index of settlement to modify")
    operation: str = Field(description="Operation: move, rename, change_type, change_population")
    name: Optional[str] = Field(default=None, description="New name")
    settlement_type: Optional[str] = Field(default=None, description="New type")
    population: Optional[int] = Field(default=None, description="New population")
    x: Optional[float] = Field(default=None, description="New X coordinate")
    y: Optional[float] = Field(default=None, description="New Y coordinate")


# Helper functions
def get_map_or_404(map_id: str):
    """Get map or raise 404 if not found."""
    with db.get_session() as session:
        result = session.execute(
            text("SELECT id FROM maps WHERE id = :map_id"),
            {"map_id": map_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Map not found")
        
        return result


# API endpoints
@router.post("/terrain", response_model=EditResponse)
async def edit_terrain(map_id: str, request: TerrainEdit):
    """
    Edit terrain features like height and biomes.
    
    Supports operations:
    - set_height: Set absolute height value
    - adjust_height: Adjust height by relative amount
    - set_biome: Change biome type
    """
    logger.info("Terrain edit requested", map_id=map_id, operation=request.operation)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    with db.get_session() as session:
        try:
            if request.operation == "set_height":
                if request.value is None:
                    raise HTTPException(status_code=400, detail="Value required for set_height operation")
                
                affected = session.execute(
                    text("""
                        UPDATE voronoi_cells 
                        SET height = :height, updated_at = NOW()
                        WHERE map_id = :map_id 
                        AND cell_index = ANY(:cell_indices)
                    """),
                    {
                        "height": request.value,
                        "map_id": map_id,
                        "cell_indices": request.cell_indices
                    }
                ).rowcount
                
            elif request.operation == "adjust_height":
                if request.value is None:
                    raise HTTPException(status_code=400, detail="Value required for adjust_height operation")
                
                affected = session.execute(
                    text("""
                        UPDATE voronoi_cells 
                        SET height = GREATEST(height + :adjustment, 0), updated_at = NOW()
                        WHERE map_id = :map_id 
                        AND cell_index = ANY(:cell_indices)
                    """),
                    {
                        "adjustment": request.value,
                        "map_id": map_id,
                        "cell_indices": request.cell_indices
                    }
                ).rowcount
                
            elif request.operation == "set_biome":
                if request.biome is None:
                    raise HTTPException(status_code=400, detail="Biome required for set_biome operation")
                
                affected = session.execute(
                    text("""
                        UPDATE voronoi_cells 
                        SET biome = :biome, updated_at = NOW()
                        WHERE map_id = :map_id 
                        AND cell_index = ANY(:cell_indices)
                    """),
                    {
                        "biome": request.biome,
                        "map_id": map_id,
                        "cell_indices": request.cell_indices
                    }
                ).rowcount
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
            
            session.commit()
            
            return EditResponse(
                success=True,
                message=f"Terrain edit completed. {affected} cells affected.",
                affected_count=affected,
                regenerate_required=affected > 0
            )
            
        except Exception as e:
            session.rollback()
            logger.error("Terrain edit failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Terrain edit failed: {str(e)}")


@router.post("/settlements", response_model=EditResponse)
async def edit_settlement(map_id: str, request: SettlementEdit):
    """
    Edit settlement properties.
    
    Supports operations:
    - move: Change settlement location
    - rename: Change settlement name
    - change_type: Change settlement type
    - change_population: Change population
    """
    logger.info("Settlement edit requested", map_id=map_id, operation=request.operation)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    with db.get_session() as session:
        try:
            # Check if settlement exists
            settlement = session.execute(
                text("""
                    SELECT settlement_index, name 
                    FROM settlements 
                    WHERE map_id = :map_id AND settlement_index = :settlement_index
                """),
                {"map_id": map_id, "settlement_index": request.settlement_index}
            ).fetchone()
            
            if not settlement:
                raise HTTPException(status_code=404, detail="Settlement not found")
            
            if request.operation == "move":
                if request.x is None or request.y is None:
                    raise HTTPException(status_code=400, detail="X and Y coordinates required for move operation")
                
                affected = session.execute(
                    text("""
                        UPDATE settlements 
                        SET geometry = ST_Point(:x, :y), updated_at = NOW()
                        WHERE map_id = :map_id AND settlement_index = :settlement_index
                    """),
                    {
                        "x": request.x,
                        "y": request.y,
                        "map_id": map_id,
                        "settlement_index": request.settlement_index
                    }
                ).rowcount
                
            elif request.operation == "rename":
                if request.name is None:
                    raise HTTPException(status_code=400, detail="Name required for rename operation")
                
                affected = session.execute(
                    text("""
                        UPDATE settlements 
                        SET name = :name, updated_at = NOW()
                        WHERE map_id = :map_id AND settlement_index = :settlement_index
                    """),
                    {
                        "name": request.name,
                        "map_id": map_id,
                        "settlement_index": request.settlement_index
                    }
                ).rowcount
                
            elif request.operation == "change_type":
                if request.settlement_type is None:
                    raise HTTPException(status_code=400, detail="Settlement type required for change_type operation")
                
                affected = session.execute(
                    text("""
                        UPDATE settlements 
                        SET settlement_type = :settlement_type, updated_at = NOW()
                        WHERE map_id = :map_id AND settlement_index = :settlement_index
                    """),
                    {
                        "settlement_type": request.settlement_type,
                        "map_id": map_id,
                        "settlement_index": request.settlement_index
                    }
                ).rowcount
                
            elif request.operation == "change_population":
                if request.population is None:
                    raise HTTPException(status_code=400, detail="Population required for change_population operation")
                
                affected = session.execute(
                    text("""
                        UPDATE settlements 
                        SET population = :population, updated_at = NOW()
                        WHERE map_id = :map_id AND settlement_index = :settlement_index
                    """),
                    {
                        "population": request.population,
                        "map_id": map_id,
                        "settlement_index": request.settlement_index
                    }
                ).rowcount
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
            
            session.commit()
            
            return EditResponse(
                success=True,
                message=f"Settlement {settlement.name} edited successfully.",
                affected_count=affected,
                regenerate_required=request.operation == "move"
            )
            
        except Exception as e:
            session.rollback()
            logger.error("Settlement edit failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Settlement edit failed: {str(e)}")


@router.get("/status", response_model=Dict[str, Any])
async def get_edit_status(map_id: str):
    """
    Get editing status and capabilities for a map.
    
    Returns information about what can be edited.
    """
    # Validate map exists
    get_map_or_404(map_id)
    
    with db.get_session() as session:
        # Get counts of editable features
        terrain_count = session.execute(
            text("SELECT COUNT(*) FROM voronoi_cells WHERE map_id = :map_id"),
            {"map_id": map_id}
        ).scalar()
        
        settlement_count = session.execute(
            text("SELECT COUNT(*) FROM settlements WHERE map_id = :map_id"),
            {"map_id": map_id}
        ).scalar()
        
        river_count = session.execute(
            text("SELECT COUNT(*) FROM rivers WHERE map_id = :map_id"),
            {"map_id": map_id}
        ).scalar()
        
        return {
            "map_id": map_id,
            "editable_features": {
                "terrain_cells": terrain_count,
                "settlements": settlement_count,
                "rivers": river_count
            },
            "available_operations": {
                "terrain": ["set_height", "adjust_height", "set_biome"],
                "settlements": ["move", "rename", "change_type", "change_population"],
                "rivers": ["reroute", "change_discharge"]
            },
            "editing_enabled": True
        }

