"""
Configuration settings for the map editor.

This module defines settings and constraints for map editing operations,
including validation rules, limits, and default values.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class EditPermissionLevel(str, Enum):
    """Permission levels for editing operations."""
    
    READ_ONLY = "read_only"
    BASIC_EDIT = "basic_edit"
    ADVANCED_EDIT = "advanced_edit"
    ADMIN = "admin"


class TerrainEditSettings(BaseModel):
    """Settings for terrain editing operations."""
    
    # Height constraints
    min_height: float = Field(default=-100.0, description="Minimum allowed height")
    max_height: float = Field(default=300.0, description="Maximum allowed height")
    max_height_change: float = Field(default=50.0, description="Maximum height change per operation")
    
    # Biome constraints
    allowed_biomes: List[str] = Field(
        default=[
            "Ocean", "Lake", "Wetland", "Desert", "Temperate Grassland",
            "Temperate Deciduous Forest", "Temperate Coniferous Forest",
            "Tropical Rainforest", "Tropical Seasonal Forest", "Alpine",
            "Tundra", "Ice", "Hot Desert", "Cold Desert", "Savanna",
            "Temperate Shrubland", "Boreal Forest", "Mangrove"
        ],
        description="List of allowed biome types"
    )
    
    # Operation limits
    max_cells_per_operation: int = Field(default=1000, description="Maximum cells per edit operation")
    require_adjacent_cells: bool = Field(default=False, description="Whether edited cells must be adjacent")


class SettlementEditSettings(BaseModel):
    """Settings for settlement editing operations."""
    
    # Population constraints
    min_population: int = Field(default=100, description="Minimum settlement population")
    max_population: int = Field(default=1000000, description="Maximum settlement population")
    
    # Settlement types
    allowed_settlement_types: List[str] = Field(
        default=["village", "town", "city", "capital", "port", "fortress"],
        description="Allowed settlement types"
    )
    
    # Placement constraints
    min_distance_between_settlements: float = Field(
        default=5.0, 
        description="Minimum distance between settlements (in map units)"
    )
    max_settlements_per_map: int = Field(default=500, description="Maximum settlements per map")
    
    # Capital constraints
    max_capitals_per_state: int = Field(default=1, description="Maximum capitals per state")
    require_capital_per_state: bool = Field(default=True, description="Whether each state needs a capital")


class CultureEditSettings(BaseModel):
    """Settings for culture editing operations."""
    
    # Culture constraints
    max_cultures_per_map: int = Field(default=50, description="Maximum cultures per map")
    min_culture_cells: int = Field(default=10, description="Minimum cells for a valid culture")
    max_culture_cells: int = Field(default=10000, description="Maximum cells per culture")
    
    # Culture types
    allowed_culture_types: List[str] = Field(
        default=["Generic", "Naval", "Nomadic", "Highland", "Lake", "Forest", "River"],
        description="Allowed culture types"
    )
    
    # Expansion constraints
    max_expansion_per_operation: int = Field(
        default=100, 
        description="Maximum cells that can be added to a culture per operation"
    )
    require_contiguous_territory: bool = Field(
        default=False, 
        description="Whether culture territory must be contiguous"
    )


class ReligionEditSettings(BaseModel):
    """Settings for religion editing operations."""
    
    # Religion constraints
    max_religions_per_map: int = Field(default=30, description="Maximum religions per map")
    min_religion_cells: int = Field(default=5, description="Minimum cells for a valid religion")
    
    # Religion types and forms
    allowed_religion_types: List[str] = Field(
        default=["Folk", "Organized", "Cult", "Heresy"],
        description="Allowed religion types"
    )
    allowed_religion_forms: List[str] = Field(
        default=[
            "Shamanism", "Animism", "Ancestor Worship", "Totemism",
            "Monotheism", "Polytheism", "Dualism", "Deism", "Pantheism"
        ],
        description="Allowed religion forms"
    )
    
    # Conversion constraints
    max_conversion_per_operation: int = Field(
        default=200, 
        description="Maximum cells that can be converted per operation"
    )
    require_cultural_compatibility: bool = Field(
        default=True, 
        description="Whether religion spread requires cultural compatibility"
    )


class StateEditSettings(BaseModel):
    """Settings for state/political editing operations."""
    
    # State constraints
    max_states_per_map: int = Field(default=100, description="Maximum states per map")
    min_state_cells: int = Field(default=20, description="Minimum cells for a valid state")
    max_state_cells: int = Field(default=5000, description="Maximum cells per state")
    
    # Political constraints
    require_capital_for_state: bool = Field(default=True, description="Whether states must have capitals")
    allow_enclaves: bool = Field(default=False, description="Whether enclaves are allowed")
    max_enclaves_per_state: int = Field(default=3, description="Maximum enclaves per state")
    
    # Merge/split constraints
    min_cells_after_split: int = Field(default=50, description="Minimum cells for states after splitting")
    require_contiguous_after_split: bool = Field(default=True, description="Whether split states must be contiguous")


class RiverEditSettings(BaseModel):
    """Settings for river editing operations."""
    
    # River constraints
    max_rivers_per_map: int = Field(default=200, description="Maximum rivers per map")
    min_river_length: float = Field(default=5.0, description="Minimum river length")
    max_river_length: float = Field(default=1000.0, description="Maximum river length")
    
    # Flow constraints
    min_discharge: float = Field(default=1.0, description="Minimum river discharge")
    max_discharge: float = Field(default=10000.0, description="Maximum river discharge")
    
    # Path constraints
    max_path_changes_per_operation: int = Field(
        default=50, 
        description="Maximum path changes per river edit operation"
    )
    require_downhill_flow: bool = Field(default=True, description="Whether rivers must flow downhill")
    allow_river_crossings: bool = Field(default=False, description="Whether rivers can cross each other")


class EditorSettings(BaseModel):
    """Main editor settings configuration."""
    
    # General settings
    enable_editing: bool = Field(default=True, description="Whether editing is enabled")
    require_authentication: bool = Field(default=False, description="Whether editing requires authentication")
    default_permission_level: EditPermissionLevel = Field(
        default=EditPermissionLevel.BASIC_EDIT, 
        description="Default permission level for users"
    )
    
    # Operation limits
    max_operations_per_batch: int = Field(default=100, description="Maximum operations per batch edit")
    max_concurrent_editors: int = Field(default=10, description="Maximum concurrent editors per map")
    
    # Validation settings
    strict_validation: bool = Field(default=True, description="Whether to use strict validation")
    auto_fix_errors: bool = Field(default=False, description="Whether to automatically fix validation errors")
    
    # Regeneration settings
    auto_regenerate_features: bool = Field(default=True, description="Whether to auto-regenerate dependent features")
    regeneration_timeout: int = Field(default=300, description="Timeout for regeneration operations (seconds)")
    
    # Feature-specific settings
    terrain: TerrainEditSettings = Field(default_factory=TerrainEditSettings)
    settlements: SettlementEditSettings = Field(default_factory=SettlementEditSettings)
    cultures: CultureEditSettings = Field(default_factory=CultureEditSettings)
    religions: ReligionEditSettings = Field(default_factory=ReligionEditSettings)
    states: StateEditSettings = Field(default_factory=StateEditSettings)
    rivers: RiverEditSettings = Field(default_factory=RiverEditSettings)
    
    # Permission mapping
    permission_constraints: Dict[EditPermissionLevel, Dict[str, Any]] = Field(
        default={
            EditPermissionLevel.READ_ONLY: {
                "allowed_operations": [],
                "max_cells_per_operation": 0
            },
            EditPermissionLevel.BASIC_EDIT: {
                "allowed_operations": ["modify", "add"],
                "max_cells_per_operation": 100,
                "allowed_features": ["settlements", "terrain"]
            },
            EditPermissionLevel.ADVANCED_EDIT: {
                "allowed_operations": ["modify", "add", "remove", "move"],
                "max_cells_per_operation": 500,
                "allowed_features": ["settlements", "terrain", "cultures", "religions"]
            },
            EditPermissionLevel.ADMIN: {
                "allowed_operations": ["modify", "add", "remove", "move", "merge", "split"],
                "max_cells_per_operation": 10000,
                "allowed_features": ["settlements", "terrain", "cultures", "religions", "states", "rivers"]
            }
        },
        description="Permission level constraints"
    )


# Default settings instance
default_editor_settings = EditorSettings()


def get_editor_settings() -> EditorSettings:
    """Get the current editor settings."""
    return default_editor_settings


def validate_edit_permission(
    operation: str, 
    feature_type: str, 
    permission_level: EditPermissionLevel,
    cell_count: int = 0
) -> bool:
    """
    Validate if an edit operation is allowed for the given permission level.
    
    Args:
        operation: The edit operation (add, remove, modify, etc.)
        feature_type: The feature being edited (terrain, settlements, etc.)
        permission_level: The user's permission level
        cell_count: Number of cells being affected
    
    Returns:
        bool: Whether the operation is allowed
    """
    settings = get_editor_settings()
    constraints = settings.permission_constraints.get(permission_level, {})
    
    # Check if operation is allowed
    allowed_operations = constraints.get("allowed_operations", [])
    if operation not in allowed_operations:
        return False
    
    # Check if feature type is allowed
    allowed_features = constraints.get("allowed_features", [])
    if feature_type not in allowed_features:
        return False
    
    # Check cell count limit
    max_cells = constraints.get("max_cells_per_operation", 0)
    if cell_count > max_cells:
        return False
    
    return True


def get_feature_constraints(feature_type: str) -> Optional[BaseModel]:
    """
    Get the constraints for a specific feature type.
    
    Args:
        feature_type: The feature type (terrain, settlements, etc.)
    
    Returns:
        BaseModel: The constraints for the feature type, or None if not found
    """
    settings = get_editor_settings()
    
    feature_settings_map = {
        "terrain": settings.terrain,
        "settlements": settings.settlements,
        "cultures": settings.cultures,
        "religions": settings.religions,
        "states": settings.states,
        "rivers": settings.rivers
    }
    
    return feature_settings_map.get(feature_type)

