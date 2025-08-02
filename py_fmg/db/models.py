"""
Database models for FMG data storage.

Enhanced schema supporting:
- Culture system with 7 culture types
- Religion system with folk and organized religions
- Enhanced settlement features with architectural elements
- Climate data storage
- Cell-level assignments for cultures and religions
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime, LargeBinary, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geometry
import uuid
from datetime import datetime

Base = declarative_base()


class Map(Base):
    """Main map table storing generation metadata."""
    
    __tablename__ = "maps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    seed = Column(String(50), nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    cells_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    generation_time_seconds = Column(Float)
    
    # Generation parameters
    config_json = Column(Text)  # JSON blob of generation parameters
    
    # Relationships
    voronoi_cells = relationship("VoronoiCell", back_populates="map", cascade="all, delete-orphan")
    cultures = relationship("Culture", back_populates="map", cascade="all, delete-orphan")
    religions = relationship("Religion", back_populates="map", cascade="all, delete-orphan")
    states = relationship("State", back_populates="map", cascade="all, delete-orphan")
    settlements = relationship("Settlement", back_populates="map", cascade="all, delete-orphan")
    rivers = relationship("River", back_populates="map", cascade="all, delete-orphan")
    biomes = relationship("BiomeRegion", back_populates="map", cascade="all, delete-orphan")

    cultures = relationship("Culture", back_populates="map", cascade="all, delete-orphan")
    religions = relationship("Religion", back_populates="map", cascade="all, delete-orphan")

    climate_data = relationship("ClimateData", back_populates="map", cascade="all, delete-orphan")
    cell_cultures = relationship("CellCulture", back_populates="map", cascade="all, delete-orphan")
    cell_religions = relationship("CellReligion", back_populates="map", cascade="all, delete-orphan")


class Culture(Base):
    """Cultural groups with geographic territories and expansion patterns."""
    
    __tablename__ = "cultures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    culture_index = Column(Integer, nullable=False)  # Original FMG culture ID
    
    name = Column(String(255), nullable=False)
    color = Column(String(7), nullable=False)  # Hex color code
    type = Column(String(50), nullable=False)  # Naval, Highland, River, Lake, Nomadic, Hunting, Generic
    expansionism = Column(Float, nullable=False, default=1.0)
    center_cell_index = Column(Integer, nullable=False)  # Reference to grid cell
    name_base = Column(Integer, nullable=False, default=0)  # Index into name bases (0-11)
    
    # Territory as polygon
    geometry = Column(Geometry("MULTIPOLYGON", srid=4326))
    
    # Statistics
    area_km2 = Column(Float)
    population = Column(Integer)
    cells_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    map = relationship("Map", back_populates="cultures")
    settlements = relationship("Settlement", back_populates="culture")
    states = relationship("State", back_populates="culture")
    origin_religions = relationship("Religion", back_populates="origin_culture")
    cell_assignments = relationship("CellCulture", back_populates="culture")
    religion_relationships = relationship("ReligionCulture", back_populates="culture")


class Religion(Base):
    """Religious systems with expansion mechanics and cultural relationships."""
    
    __tablename__ = "religions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    religion_index = Column(Integer, nullable=False)  # Original FMG religion ID
    
    name = Column(String(255), nullable=False)
    color = Column(String(7), nullable=False)
    type = Column(String(50), nullable=False)  # Folk, Organized, Cult, Heresy
    form = Column(String(100), nullable=False)  # Shamanism, Monotheism, etc.
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    center_cell_index = Column(Integer, nullable=False)
    
    # Deity information
    deity = Column(String(255))  # Supreme deity name
    
    # Expansion properties
    expansion = Column(String(20), nullable=False, default='global')  # global, state, culture
    expansionism = Column(Float, nullable=False, default=1.0)  # 0-10 scale
    code = Column(String(10))  # Abbreviated code
    
    # Relationships - stored as array of religion IDs
    origins = Column(ARRAY(Integer))  # Parent religion IDs
    
    # Territory as polygon
    geometry = Column(Geometry("MULTIPOLYGON", srid=4326))
    
    # Statistics
    area_km2 = Column(Float)
    rural_population = Column(Float, default=0)
    urban_population = Column(Float, default=0)
    cells_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    map = relationship("Map", back_populates="religions")
    origin_culture = relationship("Culture", back_populates="origin_religions")
    settlements = relationship("Settlement", back_populates="religion")
    theocratic_states = relationship("State", back_populates="state_religion")
    cell_assignments = relationship("CellReligion", back_populates="religion")
    culture_relationships = relationship("ReligionCulture", back_populates="religion")


class State(Base):
    """Political states/nations with enhanced culture and religion support."""
    
    __tablename__ = "states"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    state_index = Column(Integer, nullable=False)  # Original FMG state ID
    
    name = Column(String(255), nullable=False)
    color = Column(String(7))  # Hex color code

    culture_name = Column(String(100))

    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    religion_id = Column(UUID(as_uuid=True), ForeignKey("religions.id"), nullable=True)
    
    # Government and state properties

    government_type = Column(String(100))
    state_type = Column(String(100))  # Enhanced type including theocracies
    expansionism = Column(Float, default=1.0)
    center_cell_index = Column(Integer)
    locked = Column(Boolean, default=False)
    removed = Column(Boolean, default=False)
    
    # Territory management
    geometry = Column(Geometry("POLYGON", srid=4326))
    territory_cell_indices = Column(ARRAY(Integer))  # Alternative to complex geometry
    
    # Statistics
    area_km2 = Column(Float)
    population = Column(Integer)
    settlement_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    map = relationship("Map", back_populates="states")
    culture = relationship("Culture", back_populates="states")

    state_religion = relationship("Religion", back_populates="theocratic_states")
    settlements = relationship("Settlement", back_populates="state")


class Settlement(Base):
    """Cities, towns, and villages with enhanced cultural and religious features."""
    
    __tablename__ = "settlements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    state_id = Column(UUID(as_uuid=True), ForeignKey("states.id"), nullable=True)
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    religion_id = Column(UUID(as_uuid=True), ForeignKey("religions.id"), nullable=True)
    settlement_index = Column(Integer, nullable=False)  # Original FMG settlement ID
    
    name = Column(String(255), nullable=False)
    settlement_type = Column(String(50))  # Basic type: capital, city, town, village
    enhanced_type = Column(String(100))  # Enhanced type: Highland Fortress, Port, etc.
    population = Column(Integer)
    
    # Location
    geometry = Column(Geometry("POINT", srid=4326))
    cell_index = Column(Integer)  # Reference to original grid cell
    
    # Basic properties
    is_capital = Column(Boolean, default=False)
    is_port = Column(Boolean, default=False)

    culture_name = Column(String(100))
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    religion_id = Column(UUID(as_uuid=True), ForeignKey("religions.id"), nullable=True)

    
    # Enhanced port data
    port_feature_id = Column(Integer)  # Reference to water feature
    harbor_score = Column(Integer)  # Harbor quality 0-1
    
    # Architectural features (from enhanced settlement system)
    citadel = Column(Boolean, default=False)  # Fortified keep/castle
    plaza = Column(Boolean, default=False)    # Central market/gathering area
    walls = Column(Boolean, default=False)    # Defensive fortifications
    shanty = Column(Boolean, default=False)   # Overcrowded poor districts
    temple = Column(Boolean, default=False)   # Religious building
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    map = relationship("Map", back_populates="settlements")
    state = relationship("State", back_populates="settlements")
    culture = relationship("Culture", back_populates="settlements")
    religion = relationship("Religion", back_populates="settlements")


class River(Base):
    """River systems."""
    
    __tablename__ = "rivers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    river_index = Column(Integer, nullable=False)  # Original FMG river ID
    
    name = Column(String(255))
    
    # Geometry as linestring
    geometry = Column(Geometry("LINESTRING", srid=4326))
    
    # Properties
    length_km = Column(Float)
    discharge_m3s = Column(Float)  # Discharge in cubic meters per second
    average_width_m = Column(Float)
    
    # River hierarchy
    parent_river_id = Column(UUID(as_uuid=True), ForeignKey("rivers.id"), nullable=True)
    is_main_stem = Column(Boolean, default=False)
    
    # Source and mouth information
    source_elevation_m = Column(Float)
    mouth_elevation_m = Column(Float)
    
    # Relationships
    map = relationship("Map", back_populates="rivers")
    tributaries = relationship("River", remote_side=[id])


class BiomeRegion(Base):
    """Biome/ecological regions with enhanced 13-biome classification."""
    
    __tablename__ = "biomes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    
    biome_type = Column(String(100), nullable=False)  # Marine, Desert, Forest, etc.
    biome_index = Column(Integer, nullable=False)  # Original FMG biome ID
    biome_classification = Column(String(50))  # Enhanced classification
    
    # Classification indices
    temperature_band = Column(Integer)  # 0-25 temperature band
    moisture_band = Column(Integer)     # 0-4 moisture band
    
    # Region as polygon
    geometry = Column(Geometry("MULTIPOLYGON", srid=4326))
    
    # Properties
    area_km2 = Column(Float)
    habitability_score = Column(Integer)  # 0-100 habitability
    movement_cost = Column(Integer)  # Movement difficulty
    icons_density = Column(Integer, default=0)
    icon_weights = Column(JSONB)  # Icon type weights as JSON
    
    # Climate data
    avg_temperature_c = Column(Float)
    avg_precipitation_mm = Column(Float)
    
    # Relationships
    map = relationship("Map", back_populates="biomes")



class Culture(Base):
    """Cultural groups and regions."""
    
    __tablename__ = "cultures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    culture_index = Column(Integer, nullable=False)  # Original FMG culture ID
    
    name = Column(String(255), nullable=False)
    color = Column(String(7))  # Hex color code
    type = Column(String(50))  # Generic, Naval, Nomadic, Hunting, Highland, Lake, River
    
    # Territory as polygon
    geometry = Column(Geometry("MULTIPOLYGON", srid=4326))
    
    # Properties
    area_km2 = Column(Float)
    population = Column(Integer)
    expansionism = Column(Float, default=1.0)
    name_base = Column(Integer, default=0)  # Index into name bases
    
    # Center point
    center_geometry = Column(Geometry("POINT", srid=4326))
    center_cell_index = Column(Integer)
    
    # Relationships
    map = relationship("Map", back_populates="cultures")
    settlements = relationship("Settlement", back_populates="culture")
    states = relationship("State", back_populates="culture")


class Religion(Base):
    """Religious systems and beliefs."""
    
    __tablename__ = "religions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    religion_index = Column(Integer, nullable=False)  # Original FMG religion ID
    
    name = Column(String(255), nullable=False)
    color = Column(String(7))  # Hex color code
    type = Column(String(50))  # Folk, Organized, Cult, Heresy
    form = Column(String(100))  # Specific religious form
    
    # Associated culture and center
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    center_geometry = Column(Geometry("POINT", srid=4326))
    center_cell_index = Column(Integer)
    
    # Properties
    deity = Column(String(255))  # Supreme deity name
    expansion = Column(String(20), default="global")  # global, state, culture
    expansionism = Column(Float, default=1.0)  # 0-10, expansion competitiveness
    code = Column(String(10))  # Abbreviated code
    
    # Territory as polygon
    geometry = Column(Geometry("MULTIPOLYGON", srid=4326))
    
    # Statistics
    area_km2 = Column(Float)
    rural_population = Column(Float)
    urban_population = Column(Float)
    
    # Relationships
    map = relationship("Map", back_populates="religions")
    culture = relationship("Culture")
    settlements = relationship("Settlement", back_populates="religion")

class VoronoiCell(Base):
    """Voronoi cell geometries with heightmap data."""
    
    __tablename__ = "voronoi_cells"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    cell_index = Column(Integer, nullable=False)  # Index in the packed graph
    
    # Cell geometry as polygon
    geometry = Column(Geometry("POLYGON", srid=4326))
    
    # Height and geographic properties
    height = Column(Integer, nullable=False)  # Height value
    is_land = Column(Boolean, default=True)   # True if height >= 20
    is_coastal = Column(Boolean, default=False)  # Coastal cell
    
    # Cell center point (for easy access)
    center_x = Column(Float, nullable=False)
    center_y = Column(Float, nullable=False)
    
    # Area in map units
    area = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    map = relationship("Map", back_populates="voronoi_cells")


class ClimateData(Base):
    """Temperature and precipitation data for map cells."""
    
    __tablename__ = "climate_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    cell_index = Column(Integer, nullable=False)
    
    # Climate measurements
    temperature_c = Column(Integer)  # Temperature in Celsius (-128 to 127)
    precipitation_mm = Column(Integer)  # Annual precipitation in mm (0-65535)
    
    # Location data for reference
    latitude = Column(Float)
    altitude_m = Column(Integer)
    
    # Calculated indices for biome classification
    moisture_index = Column(Integer)     # 0-4 moisture band
    temperature_index = Column(Integer)  # 0-25 temperature band
    
    # Relationships
    map = relationship("Map", back_populates="climate_data")


class CellCulture(Base):
    """Maps cells to their culture assignments."""
    
    __tablename__ = "cell_cultures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    cell_index = Column(Integer, nullable=False)
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=False)
    
    # Statistics for this cell
    population = Column(Float, default=0)
    suitability = Column(Integer, default=0)
    
    # Relationships
    map = relationship("Map", back_populates="cell_cultures")
    culture = relationship("Culture", back_populates="cell_assignments")


class CellReligion(Base):
    """Maps cells to their religion assignments."""
    
    __tablename__ = "cell_religions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    cell_index = Column(Integer, nullable=False)
    religion_id = Column(UUID(as_uuid=True), ForeignKey("religions.id"), nullable=False)
    
    # Competition metrics
    conversion_cost = Column(Float, default=0)
    dominance_score = Column(Float, default=1.0)
    
    # Relationships
    map = relationship("Map", back_populates="cell_religions")
    religion = relationship("Religion", back_populates="cell_assignments")


class ReligionCulture(Base):
    """Many-to-many relationship between religions and cultures."""
    
    __tablename__ = "religion_cultures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    religion_id = Column(UUID(as_uuid=True), ForeignKey("religions.id"), nullable=False)
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=False)
    
    relationship_type = Column(String(50))  # origin, converted, resistant
    influence_strength = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    religion = relationship("Religion", back_populates="culture_relationships")
    culture = relationship("Culture", back_populates="religion_relationships")



class GenerationJob(Base):
    """Track map generation jobs for async processing."""
    
    __tablename__ = "generation_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=True)
    
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Request parameters
    seed = Column(String(50))
    width = Column(Float)
    height = Column(Float)
    cells_desired = Column(Integer)
    template_name = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Store intermediate data during generation
    intermediate_data = Column(LargeBinary)  # Pickled data for resumption