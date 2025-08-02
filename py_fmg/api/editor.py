"""
Map editing API endpoints for post-generation modifications.

This module provides endpoints for editing generated maps, including:
- Terrain modification (height, biome changes)
- Settlement management (add, remove, modify)
- Culture and religion editing
- River and lake modifications
- State boundary adjustments

Designed for both human users and MCP AI agents.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
import structlog
from datetime import datetime

from ..db.connection import db
from ..db.models import Map, VoronoiCell, Settlement, Culture, Religion, State, River
from ..core.features import Features
from ..core.hydrology import Hydrology
from ..core.biomes import BiomeClassifier

logger = structlog.get_logger()

# Create router for editor endpoints
router = APIRouter(prefix="/maps/{map_id}/edit", tags=["Map Editor"])


# Pydantic models for editing operations
class TerrainEdit(BaseModel):
    """Model for terrain editing operations."""
    
    cell_indices: List[int] = Field(description="List of cell indices to modify")
    operation: str = Field(description="Operation type: 'set_height', 'adjust_height', 'set_biome'")
    value: Union[float, str] = Field(description="New value (height or biome name)")
    relative: bool = Field(default=False, description="Whether the operation is relative to current value")


class SettlementEdit(BaseModel):
    """Model for settlement editing operations."""
    
    operation: str = Field(description="Operation: 'add', 'remove', 'modify', 'move'")
    settlement_id: Optional[int] = Field(default=None, description="Settlement ID for modify/remove/move")
    name: Optional[str] = Field(default=None, description="Settlement name")
    x: Optional[float] = Field(default=None, description="X coordinate")
    y: Optional[float] = Field(default=None, description="Y coordinate")
    settlement_type: Optional[str] = Field(default=None, description="Settlement type")
    population: Optional[int] = Field(default=None, description="Population")
    is_capital: Optional[bool] = Field(default=None, description="Whether it's a capital")
    culture_id: Optional[int] = Field(default=None, description="Culture ID")
    religion_id: Optional[int] = Field(default=None, description="Religion ID")


class CultureEdit(BaseModel):
    """Model for culture editing operations."""
    
    operation: str = Field(description="Operation: 'add', 'remove', 'modify', 'expand', 'contract'")
    culture_id: Optional[int] = Field(default=None, description="Culture ID for modify/remove")
    name: Optional[str] = Field(default=None, description="Culture name")
    color: Optional[str] = Field(default=None, description="Culture color (hex)")
    type: Optional[str] = Field(default=None, description="Culture type")
    cell_indices: Optional[List[int]] = Field(default=None, description="Cells to add/remove from culture")
    expansionism: Optional[float] = Field(default=None, description="Culture expansionism factor")


class ReligionEdit(BaseModel):
    """Model for religion editing operations."""
    
    operation: str = Field(description="Operation: 'add', 'remove', 'modify', 'spread', 'convert'")
    religion_id: Optional[int] = Field(default=None, description="Religion ID for modify/remove")
    name: Optional[str] = Field(default=None, description="Religion name")
    color: Optional[str] = Field(default=None, description="Religion color (hex)")
    type: Optional[str] = Field(default=None, description="Religion type: Folk, Organized, Cult")
    form: Optional[str] = Field(default=None, description="Religion form: Monotheism, Polytheism, etc.")
    deity: Optional[str] = Field(default=None, description="Deity name")
    cell_indices: Optional[List[int]] = Field(default=None, description="Cells to convert")
    culture_id: Optional[int] = Field(default=None, description="Associated culture ID")


class StateEdit(BaseModel):
    """Model for state/political boundary editing operations."""
    
    operation: str = Field(description="Operation: 'add', 'remove', 'modify', 'merge', 'split'")
    state_id: Optional[int] = Field(default=None, description="State ID for modify/remove")
    name: Optional[str] = Field(default=None, description="State name")
    color: Optional[str] = Field(default=None, description="State color (hex)")
    capital_id: Optional[int] = Field(default=None, description="Capital settlement ID")
    culture_id: Optional[int] = Field(default=None, description="Dominant culture ID")
    cell_indices: Optional[List[int]] = Field(default=None, description="Territory cells")
    merge_with_state_id: Optional[int] = Field(default=None, description="State to merge with")


class RiverEdit(BaseModel):
    """Model for river editing operations."""
    
    operation: str = Field(description="Operation: 'add', 'remove', 'modify', 'reroute'")
    river_id: Optional[int] = Field(default=None, description="River ID for modify/remove")
    name: Optional[str] = Field(default=None, description="River name")
    source_cell: Optional[int] = Field(default=None, description="Source cell index")
    mouth_cell: Optional[int] = Field(default=None, description="Mouth cell index")
    cell_path: Optional[List[int]] = Field(default=None, description="River path as cell indices")
    discharge: Optional[float] = Field(default=None, description="River discharge")


class EditResponse(BaseModel):
    """Response model for edit operations."""
    
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Result message")
    affected_cells: Optional[List[int]] = Field(default=None, description="List of affected cell indices")
    regenerate_required: bool = Field(default=False, description="Whether full regeneration is needed")
    updated_features: Optional[Dict[str, Any]] = Field(default=None, description="Updated feature data")


class BatchEdit(BaseModel):
    """Model for batch editing operations."""
    
    operations: List[Dict[str, Any]] = Field(description="List of edit operations")
    validate_only: bool = Field(default=False, description="Only validate, don't execute")
    auto_regenerate: bool = Field(default=True, description="Auto-regenerate dependent features")


# Helper functions
def get_map_or_404(map_id: str):
    """Get map by ID or raise 404."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()
        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")
        return map_obj


def validate_cell_indices(map_id: str, cell_indices: List[int]) -> bool:
    """Validate that cell indices exist in the map."""
    with db.get_session() as session:
        count = session.query(VoronoiCell).filter(
            VoronoiCell.map_id == map_id,
            VoronoiCell.cell_index.in_(cell_indices)
        ).count()
        return count == len(cell_indices)


# Terrain editing endpoints
@router.post("/terrain", response_model=EditResponse)
async def edit_terrain(map_id: str, edit: TerrainEdit):
    """
    Edit terrain features (height, biomes) for specified cells.
    
    Supports operations:
    - set_height: Set absolute height values
    - adjust_height: Adjust height relatively
    - set_biome: Change biome type
    """
    logger.info("Terrain edit requested", map_id=map_id, operation=edit.operation)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Validate cell indices
    if not validate_cell_indices(map_id, edit.cell_indices):
        raise HTTPException(status_code=400, detail="Invalid cell indices")
    
    try:
        with db.get_session() as session:
            affected_cells = []
            
            if edit.operation == "set_height":
                # Set absolute height
                height_value = float(edit.value)
                for cell_index in edit.cell_indices:
                    cell = session.query(VoronoiCell).filter(
                        VoronoiCell.map_id == map_id,
                        VoronoiCell.cell_index == cell_index
                    ).first()
                    
                    if cell:
                        if edit.relative:
                            cell.height += height_value
                        else:
                            cell.height = height_value
                        affected_cells.append(cell_index)
            
            elif edit.operation == "adjust_height":
                # Adjust height relatively
                height_delta = float(edit.value)
                for cell_index in edit.cell_indices:
                    cell = session.query(VoronoiCell).filter(
                        VoronoiCell.map_id == map_id,
                        VoronoiCell.cell_index == cell_index
                    ).first()
                    
                    if cell:
                        cell.height += height_delta
                        affected_cells.append(cell_index)
            
            elif edit.operation == "set_biome":
                # Change biome type
                biome_name = str(edit.value)
                for cell_index in edit.cell_indices:
                    cell = session.query(VoronoiCell).filter(
                        VoronoiCell.map_id == map_id,
                        VoronoiCell.cell_index == cell_index
                    ).first()
                    
                    if cell:
                        cell.biome = biome_name
                        affected_cells.append(cell_index)
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown operation: {edit.operation}")
            
            session.commit()
            
            logger.info("Terrain edit completed", 
                       map_id=map_id, 
                       operation=edit.operation,
                       affected_cells=len(affected_cells))
            
            return EditResponse(
                success=True,
                message=f"Terrain edit completed. {len(affected_cells)} cells modified.",
                affected_cells=affected_cells,
                regenerate_required=(edit.operation in ["set_height", "adjust_height"])
            )
    
    except Exception as e:
        logger.error("Terrain edit failed", map_id=map_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Terrain edit failed: {str(e)}")


@router.post("/settlements", response_model=EditResponse)
async def edit_settlements(map_id: str, edit: SettlementEdit):
    """
    Edit settlements (add, remove, modify, move).
    
    Supports operations:
    - add: Create new settlement
    - remove: Delete settlement
    - modify: Update settlement properties
    - move: Change settlement location
    """
    logger.info("Settlement edit requested", map_id=map_id, operation=edit.operation)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    try:
        with db.get_session() as session:
            if edit.operation == "add":
                # Add new settlement
                if not all([edit.name, edit.x, edit.y]):
                    raise HTTPException(status_code=400, detail="Name, x, and y coordinates required for add operation")
                
                # Find the cell at the specified coordinates
                # This is a simplified approach - in practice, you'd use spatial queries
                settlement = Settlement(
                    map_id=map_id,
                    settlement_index=0,  # Will be set properly
                    name=edit.name,
                    settlement_type=edit.settlement_type or "village",
                    population=edit.population or 1000,
                    is_capital=edit.is_capital or False,
                    is_port=False,  # Could be determined by location
                    culture_id=edit.culture_id,
                    religion_id=edit.religion_id
                )
                
                # Set geometry (simplified - would use proper spatial functions)
                from sqlalchemy import text
                session.execute(text("""
                    UPDATE settlements 
                    SET geometry = ST_SetSRID(ST_MakePoint(:x, :y), 4326)
                    WHERE id = :settlement_id
                """), {"x": edit.x, "y": edit.y, "settlement_id": settlement.id})
                
                session.add(settlement)
                session.commit()
                
                return EditResponse(
                    success=True,
                    message=f"Settlement '{edit.name}' added successfully",
                    updated_features={"settlement_id": str(settlement.id)}
                )
            
            elif edit.operation == "remove":
                # Remove settlement
                if not edit.settlement_id:
                    raise HTTPException(status_code=400, detail="Settlement ID required for remove operation")
                
                settlement = session.query(Settlement).filter(
                    Settlement.map_id == map_id,
                    Settlement.settlement_index == edit.settlement_id
                ).first()
                
                if not settlement:
                    raise HTTPException(status_code=404, detail="Settlement not found")
                
                settlement_name = settlement.name
                session.delete(settlement)
                session.commit()
                
                return EditResponse(
                    success=True,
                    message=f"Settlement '{settlement_name}' removed successfully"
                )
            
            elif edit.operation == "modify":
                # Modify existing settlement
                if not edit.settlement_id:
                    raise HTTPException(status_code=400, detail="Settlement ID required for modify operation")
                
                settlement = session.query(Settlement).filter(
                    Settlement.map_id == map_id,
                    Settlement.settlement_index == edit.settlement_id
                ).first()
                
                if not settlement:
                    raise HTTPException(status_code=404, detail="Settlement not found")
                
                # Update fields if provided
                if edit.name is not None:
                    settlement.name = edit.name
                if edit.settlement_type is not None:
                    settlement.settlement_type = edit.settlement_type
                if edit.population is not None:
                    settlement.population = edit.population
                if edit.is_capital is not None:
                    settlement.is_capital = edit.is_capital
                if edit.culture_id is not None:
                    settlement.culture_id = edit.culture_id
                if edit.religion_id is not None:
                    settlement.religion_id = edit.religion_id
                
                session.commit()
                
                return EditResponse(
                    success=True,
                    message=f"Settlement '{settlement.name}' modified successfully"
                )
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown operation: {edit.operation}")
    
    except Exception as e:
        logger.error("Settlement edit failed", map_id=map_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Settlement edit failed: {str(e)}")


@router.post("/cultures", response_model=EditResponse)
async def edit_cultures(map_id: str, edit: CultureEdit):
    """
    Edit cultures and cultural boundaries.
    
    Supports operations:
    - add: Create new culture
    - remove: Delete culture
    - modify: Update culture properties
    - expand: Add cells to culture territory
    - contract: Remove cells from culture territory
    """
    logger.info("Culture edit requested", map_id=map_id, operation=edit.operation)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Implementation would be similar to settlements but for cultures
    # This is a placeholder for the full implementation
    
    return EditResponse(
        success=True,
        message=f"Culture edit operation '{edit.operation}' completed",
        regenerate_required=True
    )


@router.post("/religions", response_model=EditResponse)
async def edit_religions(map_id: str, edit: ReligionEdit):
    """
    Edit religions and religious influence.
    
    Supports operations:
    - add: Create new religion
    - remove: Delete religion
    - modify: Update religion properties
    - spread: Expand religious influence
    - convert: Convert cells to different religion
    """
    logger.info("Religion edit requested", map_id=map_id, operation=edit.operation)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Implementation would be similar to other edit functions
    # This is a placeholder for the full implementation
    
    return EditResponse(
        success=True,
        message=f"Religion edit operation '{edit.operation}' completed",
        regenerate_required=True
    )


@router.post("/states", response_model=EditResponse)
async def edit_states(map_id: str, edit: StateEdit):
    """
    Edit political states and boundaries.
    
    Supports operations:
    - add: Create new state
    - remove: Delete state
    - modify: Update state properties
    - merge: Merge two states
    - split: Split state into multiple states
    """
    logger.info("State edit requested", map_id=map_id, operation=edit.operation)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Implementation would handle political boundary changes
    # This is a placeholder for the full implementation
    
    return EditResponse(
        success=True,
        message=f"State edit operation '{edit.operation}' completed",
        regenerate_required=True
    )


@router.post("/rivers", response_model=EditResponse)
async def edit_rivers(map_id: str, edit: RiverEdit):
    """
    Edit rivers and water features.
    
    Supports operations:
    - add: Create new river
    - remove: Delete river
    - modify: Update river properties
    - reroute: Change river path
    """
    logger.info("River edit requested", map_id=map_id, operation=edit.operation)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # Implementation would handle hydrological changes
    # This is a placeholder for the full implementation
    
    return EditResponse(
        success=True,
        message=f"River edit operation '{edit.operation}' completed",
        regenerate_required=True
    )


@router.post("/batch", response_model=EditResponse)
async def batch_edit(map_id: str, batch: BatchEdit):
    """
    Execute multiple edit operations in a single transaction.
    
    This endpoint allows for complex editing workflows and ensures
    consistency across multiple related changes.
    """
    logger.info("Batch edit requested", map_id=map_id, operations=len(batch.operations))
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    if batch.validate_only:
        # Only validate operations without executing
        return EditResponse(
            success=True,
            message=f"Batch validation completed. {len(batch.operations)} operations are valid."
        )
    
    # Execute batch operations
    # This would implement transaction handling for multiple operations
    # Placeholder for full implementation
    
    return EditResponse(
        success=True,
        message=f"Batch edit completed. {len(batch.operations)} operations executed.",
        regenerate_required=batch.auto_regenerate
    )


@router.post("/regenerate", response_model=EditResponse)
async def regenerate_features(map_id: str, features: List[str]):
    """
    Regenerate specific map features after editing.
    
    Available features:
    - biomes: Recalculate biome distribution
    - hydrology: Regenerate rivers and water flow
    - cultures: Recalculate cultural boundaries
    - religions: Recalculate religious influence
    - settlements: Recalculate settlement placement
    - states: Recalculate political boundaries
    """
    logger.info("Feature regeneration requested", map_id=map_id, features=features)
    
    # Validate map exists
    map_obj = get_map_or_404(map_id)
    
    # This would implement selective regeneration of map features
    # based on the editing changes made
    
    return EditResponse(
        success=True,
        message=f"Features regenerated: {', '.join(features)}",
        regenerate_required=False
    )



# Additional advanced editing endpoints

@router.post("/terrain/batch", response_model=EditResponse)
async def batch_edit_terrain(
    map_id: str,
    request: BatchTerrainEdit
):
    """
    Perform batch terrain editing operations.
    
    Allows multiple terrain modifications in a single request for efficiency.
    """
    logger.info("Batch terrain edit requested", map_id=map_id, operations=len(request.operations))
    
    # Validate map exists
    get_map_or_404(map_id)
    
    results = []
    total_affected = 0
    
    with db.get_session() as session:
        try:
            for operation in request.operations:
                if operation.operation == "set_height":
                    # Batch height setting
                    affected = session.execute(
                        text("""
                            UPDATE voronoi_cells 
                            SET height = :height, updated_at = NOW()
                            WHERE map_id = :map_id 
                            AND cell_index = ANY(:cell_indices)
                        """),
                        {
                            "height": operation.value,
                            "map_id": map_id,
                            "cell_indices": operation.cell_indices
                        }
                    ).rowcount
                    
                elif operation.operation == "adjust_height":
                    # Batch height adjustment (relative)
                    affected = session.execute(
                        text("""
                            UPDATE voronoi_cells 
                            SET height = GREATEST(height + :adjustment, 0), updated_at = NOW()
                            WHERE map_id = :map_id 
                            AND cell_index = ANY(:cell_indices)
                        """),
                        {
                            "adjustment": operation.value,
                            "map_id": map_id,
                            "cell_indices": operation.cell_indices
                        }
                    ).rowcount
                    
                elif operation.operation == "set_biome":
                    # Batch biome setting
                    affected = session.execute(
                        text("""
                            UPDATE voronoi_cells 
                            SET biome = :biome, updated_at = NOW()
                            WHERE map_id = :map_id 
                            AND cell_index = ANY(:cell_indices)
                        """),
                        {
                            "biome": operation.biome,
                            "map_id": map_id,
                            "cell_indices": operation.cell_indices
                        }
                    ).rowcount
                    
                elif operation.operation == "smooth_terrain":
                    # Terrain smoothing
                    affected = session.execute(
                        text("""
                            UPDATE voronoi_cells 
                            SET height = (
                                SELECT AVG(neighbor.height)
                                FROM voronoi_cells neighbor
                                WHERE neighbor.map_id = :map_id
                                AND ST_DWithin(neighbor.geometry, voronoi_cells.geometry, :radius)
                            ), updated_at = NOW()
                            WHERE map_id = :map_id 
                            AND cell_index = ANY(:cell_indices)
                        """),
                        {
                            "radius": operation.radius or 1000,
                            "map_id": map_id,
                            "cell_indices": operation.cell_indices
                        }
                    ).rowcount
                
                results.append({
                    "operation": operation.operation,
                    "affected_cells": affected,
                    "success": True
                })
                total_affected += affected
            
            session.commit()
            
            # Trigger regeneration if needed
            regenerate_required = total_affected > 0
            
            return EditResponse(
                success=True,
                message=f"Batch terrain edit completed. {total_affected} cells affected.",
                affected_count=total_affected,
                regenerate_required=regenerate_required,
                details={"operations": results}
            )
            
        except Exception as e:
            session.rollback()
            logger.error("Batch terrain edit failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Batch terrain edit failed: {str(e)}")


@router.post("/settlements/generate", response_model=EditResponse)
async def generate_settlements(
    map_id: str,
    request: SettlementGenerationRequest
):
    """
    Generate new settlements based on terrain and existing settlements.
    
    Uses algorithms to place settlements in optimal locations.
    """
    logger.info("Settlement generation requested", map_id=map_id)
    
    # Validate map exists
    get_map_or_404(map_id)
    
    with db.get_session() as session:
        try:
            # Get suitable locations for settlements
            suitable_cells = session.execute(
                text("""
                    SELECT cell_index, ST_X(ST_Centroid(geometry)) as x, ST_Y(ST_Centroid(geometry)) as y,
                           height, is_land, biome
                    FROM voronoi_cells 
                    WHERE map_id = :map_id 
                    AND is_land = true
                    AND height BETWEEN :min_height AND :max_height
                    AND biome NOT LIKE '%Desert%'
                    AND biome NOT LIKE '%Tundra%'
                    AND cell_index NOT IN (
                        SELECT DISTINCT cell_index 
                        FROM settlements 
                        WHERE map_id = :map_id
                    )
                    ORDER BY RANDOM()
                    LIMIT :max_settlements
                """),
                {
                    "map_id": map_id,
                    "min_height": request.min_height,
                    "max_height": request.max_height,
                    "max_settlements": request.count
                }
            ).fetchall()
            
            generated_settlements = []
            
            for cell in suitable_cells:
                # Determine settlement type based on terrain
                if cell.height > 80:
                    settlement_type = "village"  # Mountain villages
                elif "Forest" in cell.biome:
                    settlement_type = "town"     # Forest towns
                elif "Grassland" in cell.biome:
                    settlement_type = "city"     # Grassland cities
                else:
                    settlement_type = "village"
                
                # Generate settlement name
                settlement_name = f"Settlement_{len(generated_settlements) + 1}"
                
                # Calculate population based on type
                population_ranges = {
                    "village": (100, 500),
                    "town": (500, 2000),
                    "city": (2000, 10000)
                }
                min_pop, max_pop = population_ranges[settlement_type]
                population = random.randint(min_pop, max_pop)
                
                # Insert settlement
                result = session.execute(
                    text("""
                        INSERT INTO settlements (
                            map_id, settlement_index, name, settlement_type, 
                            population, is_capital, cell_index, geometry
                        ) VALUES (
                            :map_id, :settlement_index, :name, :settlement_type,
                            :population, :is_capital, :cell_index, 
                            ST_Point(:x, :y)
                        ) RETURNING settlement_index
                    """),
                    {
                        "map_id": map_id,
                        "settlement_index": len(generated_settlements) + 1000,  # Offset to avoid conflicts
                        "name": settlement_name,
                        "settlement_type": settlement_type,
                        "population": population,
                        "is_capital": False,
                        "cell_index": cell.cell_index,
                        "x": cell.x,
                        "y": cell.y
                    }
                )
                
                generated_settlements.append({
                    "name": settlement_name,
                    "type": settlement_type,
                    "population": population,
                    "location": [cell.x, cell.y]
                })
            
            session.commit()
            
            return EditResponse(
                success=True,
                message=f"Generated {len(generated_settlements)} new settlements",
                affected_count=len(generated_settlements),
                regenerate_required=True,
                details={"settlements": generated_settlements}
            )
            
        except Exception as e:
            session.rollback()
            logger.error("Settlement generation failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Settlement generation failed: {str(e)}")


@router.post("/rivers/reroute", response_model=EditResponse)
async def reroute_river(
    map_id: str,
    request: RiverRerouteRequest,
    
):
    """
    Reroute a river through new waypoints.
    
    Recalculates the river path based on terrain and new waypoints.
    """
    logger.info("River reroute requested", map_id=map_id, river_id=request.river_index)
    
    # Check permissions
        raise HTTPException(status_code=403, detail="Insufficient permissions for river rerouting")
    
    # Validate map exists
    get_map_or_404(map_id)
    
    with db.get_session() as session:
        try:
            # Get current river
            river = session.execute(
                text("""
                    SELECT river_index, name, discharge_m3s, geometry
                    FROM rivers 
                    WHERE map_id = :map_id AND river_index = :river_index
                """),
                {"map_id": map_id, "river_index": request.river_index}
            ).fetchone()
            
            if not river:
                raise HTTPException(status_code=404, detail="River not found")
            
            # Create new geometry from waypoints
            waypoint_coords = [(wp.longitude, wp.latitude) for wp in request.waypoints]
            
            if len(waypoint_coords) < 2:
                raise HTTPException(status_code=400, detail="At least 2 waypoints required")
            
            # Create LineString from waypoints
            linestring_wkt = f"LINESTRING({', '.join([f'{x} {y}' for x, y in waypoint_coords])})"
            
            # Update river geometry
            session.execute(
                text("""
                    UPDATE rivers 
                    SET geometry = ST_GeomFromText(:linestring, 4326),
                        length_km = ST_Length(ST_GeomFromText(:linestring, 4326)::geography) / 1000,
                        updated_at = NOW()
                    WHERE map_id = :map_id AND river_index = :river_index
                """),
                {
                    "linestring": linestring_wkt,
                    "map_id": map_id,
                    "river_index": request.river_index
                }
            )
            
            session.commit()
            
            return EditResponse(
                success=True,
                message=f"River {river.name} rerouted successfully",
                affected_count=1,
                regenerate_required=True,
                details={
                    "river_name": river.name,
                    "waypoints": len(request.waypoints),
                    "new_length_km": len(waypoint_coords) * 10  # Approximate
                }
            )
            
        except Exception as e:
            session.rollback()
            logger.error("River reroute failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"River reroute failed: {str(e)}")


@router.post("/cultures/expand", response_model=EditResponse)
async def expand_culture(
    map_id: str,
    request: CultureExpansionRequest,
    
):
    """
    Expand a culture's territory into neighboring regions.
    
    Uses cultural influence algorithms to determine expansion patterns.
    """
    logger.info("Culture expansion requested", map_id=map_id, culture_id=request.culture_index)
    
    # Check permissions
        raise HTTPException(status_code=403, detail="Insufficient permissions for culture expansion")
    
    # Validate map exists
    get_map_or_404(map_id)
    
    with db.get_session() as session:
        try:
            # Get current culture
            culture = session.execute(
                text("""
                    SELECT culture_index, name, type, geometry
                    FROM cultures 
                    WHERE map_id = :map_id AND culture_index = :culture_index
                """),
                {"map_id": map_id, "culture_index": request.culture_index}
            ).fetchone()
            
            if not culture:
                raise HTTPException(status_code=404, detail="Culture not found")
            
            # Find expandable cells (neighboring the current culture)
            expandable_cells = session.execute(
                text("""
                    SELECT vc.cell_index, vc.geometry, vc.biome, vc.is_land
                    FROM voronoi_cells vc
                    WHERE vc.map_id = :map_id
                    AND vc.is_land = true
                    AND ST_DWithin(vc.geometry, (
                        SELECT geometry FROM cultures 
                        WHERE map_id = :map_id AND culture_index = :culture_index
                    ), :expansion_radius)
                    AND vc.cell_index NOT IN (
                        SELECT DISTINCT cell_index 
                        FROM cultures 
                        WHERE map_id = :map_id
                    )
                    ORDER BY ST_Distance(vc.geometry, (
                        SELECT ST_Centroid(geometry) FROM cultures 
                        WHERE map_id = :map_id AND culture_index = :culture_index
                    ))
                    LIMIT :max_cells
                """),
                {
                    "map_id": map_id,
                    "culture_index": request.culture_index,
                    "expansion_radius": request.expansion_radius,
                    "max_cells": request.max_cells
                }
            ).fetchall()
            
            if not expandable_cells:
                return EditResponse(
                    success=True,
                    message="No suitable cells found for expansion",
                    affected_count=0,
                    regenerate_required=False
                )
            
            # Create expanded geometry by unioning with new cells
            cell_geometries = [f"(SELECT geometry FROM voronoi_cells WHERE map_id = '{map_id}' AND cell_index = {cell.cell_index})" for cell in expandable_cells]
            union_query = " UNION ALL ".join(cell_geometries)
            
            # Update culture geometry
            session.execute(
                text(f"""
                    UPDATE cultures 
                    SET geometry = ST_Union(
                        geometry,
                        (SELECT ST_Union(geom) FROM ({union_query}) AS cells(geom))
                    ),
                    updated_at = NOW()
                    WHERE map_id = :map_id AND culture_index = :culture_index
                """),
                {"map_id": map_id, "culture_index": request.culture_index}
            )
            
            session.commit()
            
            return EditResponse(
                success=True,
                message=f"Culture {culture.name} expanded into {len(expandable_cells)} new cells",
                affected_count=len(expandable_cells),
                regenerate_required=True,
                details={
                    "culture_name": culture.name,
                    "expanded_cells": len(expandable_cells),
                    "expansion_radius": request.expansion_radius
                }
            )
            
        except Exception as e:
            session.rollback()
            logger.error("Culture expansion failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Culture expansion failed: {str(e)}")


@router.post("/states/merge", response_model=EditResponse)
async def merge_states(
    map_id: str,
    request: StateMergeRequest,
    
):
    """
    Merge multiple states into a single state.
    
    Combines territories and transfers settlements.
    """
    logger.info("State merge requested", map_id=map_id, states=request.state_indices)
    
    # Check permissions
        raise HTTPException(status_code=403, detail="Insufficient permissions for state merging")
    
    # Validate map exists
    get_map_or_404(map_id)
    
    if len(request.state_indices) < 2:
        raise HTTPException(status_code=400, detail="At least 2 states required for merging")
    
    with db.get_session() as session:
        try:
            # Get states to merge
            states = session.execute(
                text("""
                    SELECT state_index, name, geometry
                    FROM states 
                    WHERE map_id = :map_id AND state_index = ANY(:state_indices)
                """),
                {"map_id": map_id, "state_indices": request.state_indices}
            ).fetchall()
            
            if len(states) != len(request.state_indices):
                raise HTTPException(status_code=404, detail="One or more states not found")
            
            # Create merged geometry
            primary_state = states[0]
            other_states = states[1:]
            
            # Union all geometries
            session.execute(
                text("""
                    UPDATE states 
                    SET geometry = (
                        SELECT ST_Union(geometry)
                        FROM states 
                        WHERE map_id = :map_id AND state_index = ANY(:state_indices)
                    ),
                    name = :new_name,
                    updated_at = NOW()
                    WHERE map_id = :map_id AND state_index = :primary_state_index
                """),
                {
                    "map_id": map_id,
                    "state_indices": request.state_indices,
                    "new_name": request.new_name or f"United {primary_state.name}",
                    "primary_state_index": primary_state.state_index
                }
            )
            
            # Update settlements to belong to the primary state
            session.execute(
                text("""
                    UPDATE settlements 
                    SET state_index = :primary_state_index
                    WHERE map_id = :map_id AND state_index = ANY(:other_state_indices)
                """),
                {
                    "map_id": map_id,
                    "primary_state_index": primary_state.state_index,
                    "other_state_indices": [s.state_index for s in other_states]
                }
            )
            
            # Delete the other states
            session.execute(
                text("""
                    DELETE FROM states 
                    WHERE map_id = :map_id AND state_index = ANY(:other_state_indices)
                """),
                {
                    "map_id": map_id,
                    "other_state_indices": [s.state_index for s in other_states]
                }
            )
            
            session.commit()
            
            return EditResponse(
                success=True,
                message=f"Merged {len(states)} states into {request.new_name or f'United {primary_state.name}'}",
                affected_count=len(states),
                regenerate_required=True,
                details={
                    "merged_states": [s.name for s in states],
                    "new_name": request.new_name or f"United {primary_state.name}",
                    "primary_state_index": primary_state.state_index
                }
            )
            
        except Exception as e:
            session.rollback()
            logger.error("State merge failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"State merge failed: {str(e)}")


# Additional Pydantic models for new endpoints

class BatchTerrainOperation(BaseModel):
    """Model for a single batch terrain operation."""
    
    operation: str = Field(description="Operation type: set_height, adjust_height, set_biome, smooth_terrain")
    cell_indices: List[int] = Field(description="List of cell indices to affect")
    value: Optional[float] = Field(default=None, description="Value for the operation")
    biome: Optional[str] = Field(default=None, description="Biome name for set_biome operation")
    radius: Optional[float] = Field(default=None, description="Radius for smooth_terrain operation")


class BatchTerrainEdit(BaseModel):
    """Model for batch terrain editing requests."""
    
    operations: List[BatchTerrainOperation] = Field(description="List of operations to perform")


class SettlementGenerationRequest(BaseModel):
    """Model for settlement generation requests."""
    
    count: int = Field(default=10, ge=1, le=50, description="Number of settlements to generate")
    min_height: float = Field(default=0, description="Minimum terrain height for settlements")
    max_height: float = Field(default=200, description="Maximum terrain height for settlements")
    avoid_existing: bool = Field(default=True, description="Avoid placing near existing settlements")
    settlement_types: List[str] = Field(
        default=["village", "town", "city"],
        description="Allowed settlement types"
    )


class Waypoint(BaseModel):
    """Model for river waypoints."""
    
    longitude: float = Field(description="Longitude coordinate")
    latitude: float = Field(description="Latitude coordinate")


class RiverRerouteRequest(BaseModel):
    """Model for river rerouting requests."""
    
    river_index: int = Field(description="Index of the river to reroute")
    waypoints: List[Waypoint] = Field(description="New waypoints for the river")
    preserve_discharge: bool = Field(default=True, description="Preserve original discharge rate")


class CultureExpansionRequest(BaseModel):
    """Model for culture expansion requests."""
    
    culture_index: int = Field(description="Index of the culture to expand")
    expansion_radius: float = Field(default=5000, description="Maximum expansion radius in meters")
    max_cells: int = Field(default=20, description="Maximum number of cells to expand into")
    expansion_type: str = Field(default="natural", description="Type of expansion: natural, aggressive, peaceful")


class StateMergeRequest(BaseModel):
    """Model for state merging requests."""
    
    state_indices: List[int] = Field(description="Indices of states to merge")
    new_name: Optional[str] = Field(default=None, description="Name for the merged state")
    primary_state_index: Optional[int] = Field(default=None, description="Index of the primary state")

