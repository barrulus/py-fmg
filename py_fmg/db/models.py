"""Database models for FMG data storage."""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from geoalchemy2 import Geometry
import uuid
from datetime import datetime

Base = declarative_base()


class Map(Base):
    """Main map table storing generation metadata."""
    
    __tablename__ = "maps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    seed = Column(String(50), nullable=False)  # Legacy field for backwards compatibility
    grid_seed = Column(String(50), nullable=False)  # Seed for Voronoi grid generation
    map_seed = Column(String(50), nullable=False)   # Seed for heightmap generation
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    cells_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    generation_time_seconds = Column(Float)
    
    # Generation parameters
    config_json = Column(Text)  # JSON blob of generation parameters
    
    # Relationships
    states = relationship("State", back_populates="map", cascade="all, delete-orphan")
    settlements = relationship("Settlement", back_populates="map", cascade="all, delete-orphan")
    rivers = relationship("River", back_populates="map", cascade="all, delete-orphan")
    biomes = relationship("BiomeRegion", back_populates="map", cascade="all, delete-orphan")
    cultures = relationship("Culture", back_populates="map", cascade="all, delete-orphan")
    religions = relationship("Religion", back_populates="map", cascade="all, delete-orphan")


class State(Base):
    """Political states/nations."""
    
    __tablename__ = "states"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    state_index = Column(Integer, nullable=False)  # Original FMG state ID
    
    name = Column(String(255), nullable=False)
    color = Column(String(7))  # Hex color code
    culture_name = Column(String(100))
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    government_type = Column(String(100))
    
    # Territory as polygon
    geometry = Column(Geometry("POLYGON", srid=4326))
    
    # Statistics
    area_km2 = Column(Float)
    population = Column(Integer)
    settlement_count = Column(Integer)
    
    # Relationships
    map = relationship("Map", back_populates="states")
    culture = relationship("Culture", back_populates="states")
    settlements = relationship("Settlement", back_populates="state")


class Settlement(Base):
    """Cities, towns, and villages."""
    
    __tablename__ = "settlements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    state_id = Column(UUID(as_uuid=True), ForeignKey("states.id"), nullable=True)
    settlement_index = Column(Integer, nullable=False)  # Original FMG settlement ID
    
    name = Column(String(255), nullable=False)
    settlement_type = Column(String(50))  # capital, city, town, village
    population = Column(Integer)
    
    # Location
    geometry = Column(Geometry("POINT", srid=4326))
    cell_index = Column(Integer)  # Reference to original grid cell
    
    # Properties
    is_capital = Column(Boolean, default=False)
    is_port = Column(Boolean, default=False)
    culture_name = Column(String(100))
    culture_id = Column(UUID(as_uuid=True), ForeignKey("cultures.id"), nullable=True)
    religion_id = Column(UUID(as_uuid=True), ForeignKey("religions.id"), nullable=True)
    
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
    """Biome/ecological regions."""
    
    __tablename__ = "biomes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=False)
    
    biome_type = Column(String(100), nullable=False)  # Marine, Desert, Forest, etc.
    biome_index = Column(Integer, nullable=False)  # Original FMG biome ID
    
    # Region as polygon
    geometry = Column(Geometry("MULTIPOLYGON", srid=4326))
    
    # Properties
    area_km2 = Column(Float)
    habitability_score = Column(Integer)  # 0-100 habitability
    movement_cost = Column(Integer)  # Movement difficulty
    
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


class GenerationJob(Base):
    """Track map generation jobs for async processing."""
    
    __tablename__ = "generation_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    map_id = Column(UUID(as_uuid=True), ForeignKey("maps.id"), nullable=True)
    
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Request parameters
    seed = Column(String(50))  # Legacy field for backwards compatibility
    grid_seed = Column(String(50))  # Seed for Voronoi grid generation
    map_seed = Column(String(50))   # Seed for heightmap generation
    width = Column(Float)
    height = Column(Float)
    cells_desired = Column(Integer)
    template_name = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Store intermediate data during generation
    intermediate_data = Column(LargeBinary)  # Pickled data for resumption