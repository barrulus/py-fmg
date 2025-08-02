"""FastAPI main application."""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from typing import Optional
import structlog
import logging

import uuid
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import settings
from ..core.biomes import BiomeClassifier
from ..core.cell_packing import regraph
from ..core.climate import Climate
from ..core.cultures import CultureGenerator
from ..core.features import Features

from ..core.climate import Climate, ClimateOptions, MapCoordinates
from ..core.hydrology import Hydrology, HydrologyOptions
from ..core.biomes import BiomeClassifier, BiomeOptions

# Set up standard logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
)

from ..core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from ..core.hydrology import Hydrology
from ..core.name_generator import NameGenerator
from ..core.settlements import Settlements
from ..core.voronoi_graph import GridConfig, generate_voronoi_graph
from ..db.connection import db
from ..db.models import (
    GenerationJob,
    Map,
    VoronoiCell,
    ClimateData,
    River,
    Culture,
    CellCulture,
    BiomeRegion,
    Religion,
    CellReligion,
    Settlement,
)

# Then configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Fantasy Map Generator API",
    description="Python port of Azgaar's Fantasy Map Generator",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class MapGenerationRequest(BaseModel):
    """Request to generate a new map."""

    seed: Optional[str] = Field(
        None, description="Random seed for reproducible generation"
    )
    width: float = Field(800, ge=100, le=2000, description="Map width")
    height: float = Field(600, ge=100, le=2000, description="Map height")
    cells_desired: int = Field(
        10000, ge=1000, le=50000, description="Target number of cells"
    )
    template_name: str = Field("default", description="Heightmap template name")
    map_name: Optional[str] = Field(None, description="Custom map name")


class JobResponse(BaseModel):
    """Response with job information."""

    job_id: str
    status: str
    progress_percent: int
    message: str
    map_id: Optional[str] = None
    error_message: Optional[str] = None


class MapSummary(BaseModel):
    """Summary information about a generated map."""

    id: str
    name: str
    seed: str
    width: float
    height: float
    cells_count: int
    created_at: datetime
    generation_time_seconds: Optional[float]


class BiomeStatistics(BaseModel):
    """Biome distribution statistics for a map."""

    biome_name: str
    cell_count: int
    percentage: float
    avg_temperature: Optional[float] = None
    avg_precipitation: Optional[float] = None


class RiverInfo(BaseModel):
    """Information about a river."""

    id: int
    length: float
    flow: float
    source_cell: int
    mouth_cell: int
    cell_count: int


class MapStatistics(BaseModel):
    """Comprehensive statistics about a generated map."""

    map_id: str
    total_cells: int
    land_cells: int
    water_cells: int
    rivers_count: int
    lakes_count: int
    biome_distribution: List[BiomeStatistics]
    major_rivers: List[RiverInfo]
    temperature_range: Tuple[float, float]
    precipitation_range: Tuple[float, float]


class ClimateData(BaseModel):
    """Climate data for a specific cell or region."""

    cell_index: int
    temperature: int
    precipitation: int
    biome: str
    height: int


class CultureInfo(BaseModel):
    """Information about a culture."""

    id: int
    name: str
    color: str
    type: str
    area_km2: float
    population: int
    expansionism: float
    center_cell: int


class ReligionInfo(BaseModel):
    """Information about a religion."""

    id: int
    name: str
    color: str
    type: str
    form: str
    deity: Optional[str]
    expansion: str
    expansionism: float
    area_km2: float
    rural_population: float
    urban_population: float


class SettlementInfo(BaseModel):
    """Information about a settlement."""

    id: int
    name: str
    type: str
    population: int
    is_capital: bool
    is_port: bool
    culture_name: Optional[str]
    state_name: Optional[str]
    religion_name: Optional[str]


# Event handlers
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize database on startup."""
    logger.info("Starting Fantasy Map Generator API")
    db.initialize()
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    logger.info("Shutting down Fantasy Map Generator API")


# API endpoints
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Fantasy Map Generator API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        # Test database connection
        from sqlalchemy import text

        with db.get_session() as session:
            session.execute(text("SELECT 1"))

        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/maps/generate", response_model=JobResponse)
async def generate_map(
    request: MapGenerationRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """
    Start map generation job.

    Returns immediately with job ID. Use /maps/status/{job_id} to check status.
    """
    logger.info("Map generation requested", request=request.dict())

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Generate seed if not provided
    seed = request.seed or str(uuid.uuid4())[:8]

    # Create job record
    with db.get_session() as session:
        job = GenerationJob(
            id=job_id,
            seed=seed,
            width=request.width,
            height=request.height,
            cells_desired=request.cells_desired,
            template_name=request.template_name,
            status="pending",
        )
        session.add(job)
        session.commit()

    # Start background generation
    background_tasks.add_task(run_map_generation, job_id, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        progress_percent=0,
        message="Map generation job started",
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str) -> JobResponse:
    """Get status of a map generation job."""
    with db.get_session() as session:
        job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobResponse(
            job_id=str(job.id),
            status=job.status,
            progress_percent=job.progress_percent,
            message=f"Job {job.status}",
            map_id=str(job.map_id) if job.map_id else None,
            error_message=job.error_message,
        )


@app.get("/maps/status/{job_id}", response_model=JobResponse)
async def get_map_generation_status(job_id: str) -> JobResponse:
    """Get status of a map generation job (alternative endpoint path as specified in issue #11)."""
    with db.get_session() as session:
        job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobResponse(
            job_id=str(job.id),
            status=job.status,
            progress_percent=job.progress_percent,
            message=f"Job {job.status}",
            map_id=str(job.map_id) if job.map_id else None,
            error_message=job.error_message,
        )


@app.get("/maps", response_model=list[MapSummary])
async def list_maps() -> list[MapSummary]:
    """List all generated maps."""
    with db.get_session() as session:
        maps = session.query(Map).order_by(Map.created_at.desc()).all()

        return [
            MapSummary(
                id=str(map.id),
                name=map.name,
                seed=map.seed,
                width=map.width,
                height=map.height,
                cells_count=map.cells_count,
                created_at=map.created_at,
                generation_time_seconds=map.generation_time_seconds,
            )
            for map in maps
        ]


@app.get("/maps/{map_id}", response_model=MapSummary)
async def get_map(map_id: str) -> MapSummary:
    """Get map details."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        return MapSummary(
            id=str(map_obj.id),
            name=map_obj.name,
            seed=map_obj.seed,
            width=map_obj.width,
            height=map_obj.height,
            cells_count=map_obj.cells_count,
            created_at=map_obj.created_at,
            generation_time_seconds=map_obj.generation_time_seconds,
        )


@app.get("/maps/{map_id}/statistics", response_model=MapStatistics)
async def get_map_statistics(map_id: str):
    """Get comprehensive statistics for a map."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        # For now, return mock statistics since we don't store the graph data yet
        # In a full implementation, this would load the graph data and calculate real statistics

        mock_biome_stats = [
            BiomeStatistics(
                biome_name="Ocean",
                cell_count=3000,
                percentage=30.0,
                avg_temperature=15.0,
                avg_precipitation=100.0,
            ),
            BiomeStatistics(
                biome_name="Temperate Forest",
                cell_count=2500,
                percentage=25.0,
                avg_temperature=12.0,
                avg_precipitation=120.0,
            ),
            BiomeStatistics(
                biome_name="Grassland",
                cell_count=2000,
                percentage=20.0,
                avg_temperature=18.0,
                avg_precipitation=80.0,
            ),
            BiomeStatistics(
                biome_name="Desert",
                cell_count=1500,
                percentage=15.0,
                avg_temperature=25.0,
                avg_precipitation=20.0,
            ),
            BiomeStatistics(
                biome_name="Mountains",
                cell_count=1000,
                percentage=10.0,
                avg_temperature=5.0,
                avg_precipitation=150.0,
            ),
        ]

        mock_rivers = [
            RiverInfo(
                id=1,
                length=250.5,
                flow=150.0,
                source_cell=100,
                mouth_cell=5000,
                cell_count=45,
            ),
            RiverInfo(
                id=2,
                length=180.2,
                flow=80.0,
                source_cell=200,
                mouth_cell=4800,
                cell_count=32,
            ),
            RiverInfo(
                id=3,
                length=120.8,
                flow=45.0,
                source_cell=300,
                mouth_cell=4500,
                cell_count=22,
            ),
        ]

        return MapStatistics(
            map_id=str(map_obj.id),
            total_cells=map_obj.cells_count,
            land_cells=int(map_obj.cells_count * 0.7),  # Mock: 70% land
            water_cells=int(map_obj.cells_count * 0.3),  # Mock: 30% water
            rivers_count=15,  # Mock
            lakes_count=5,  # Mock
            biome_distribution=mock_biome_stats,
            major_rivers=mock_rivers,
            temperature_range=(-10.0, 35.0),  # Mock temperature range
            precipitation_range=(10.0, 200.0),  # Mock precipitation range
        )


@app.get("/maps/{map_id}/climate/{cell_index}", response_model=ClimateData)
async def get_cell_climate(map_id: str, cell_index: int):
    """Get climate data for a specific cell."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        if cell_index < 0 or cell_index >= map_obj.cells_count:
            raise HTTPException(status_code=400, detail="Invalid cell index")

        # For now, return mock data
        # In a full implementation, this would load the actual graph data

        return ClimateData(
            cell_index=cell_index,
            temperature=15,  # Mock temperature
            precipitation=100,  # Mock precipitation
            biome="Temperate Forest",  # Mock biome
            height=45,  # Mock height
        )


@app.get("/maps/{map_id}/biomes", response_model=List[BiomeStatistics])
async def get_map_biomes(map_id: str):
    """Get biome distribution for a map."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        # For now, return mock biome data
        # In a full implementation, this would load the actual biome data from the graph

        return [
            BiomeStatistics(
                biome_name="Ocean",
                cell_count=3000,
                percentage=30.0,
                avg_temperature=15.0,
                avg_precipitation=100.0,
            ),
            BiomeStatistics(
                biome_name="Temperate Deciduous Forest",
                cell_count=2500,
                percentage=25.0,
                avg_temperature=12.0,
                avg_precipitation=120.0,
            ),
            BiomeStatistics(
                biome_name="Temperate Grassland",
                cell_count=2000,
                percentage=20.0,
                avg_temperature=18.0,
                avg_precipitation=80.0,
            ),
            BiomeStatistics(
                biome_name="Desert",
                cell_count=1500,
                percentage=15.0,
                avg_temperature=25.0,
                avg_precipitation=20.0,
            ),
            BiomeStatistics(
                biome_name="Alpine",
                cell_count=1000,
                percentage=10.0,
                avg_temperature=5.0,
                avg_precipitation=150.0,
            ),
        ]


@app.get("/maps/{map_id}/rivers", response_model=List[RiverInfo])
async def get_map_rivers(map_id: str):
    """Get river information for a map."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        # For now, return mock river data
        # In a full implementation, this would load the actual river data from the graph

        return [
            RiverInfo(
                id=1,
                length=250.5,
                flow=150.0,
                source_cell=100,
                mouth_cell=5000,
                cell_count=45,
            ),
            RiverInfo(
                id=2,
                length=180.2,
                flow=80.0,
                source_cell=200,
                mouth_cell=4800,
                cell_count=32,
            ),
            RiverInfo(
                id=3,
                length=120.8,
                flow=45.0,
                source_cell=300,
                mouth_cell=4500,
                cell_count=22,
            ),
            RiverInfo(
                id=4,
                length=95.3,
                flow=35.0,
                source_cell=400,
                mouth_cell=4200,
                cell_count=18,
            ),
            RiverInfo(
                id=5,
                length=75.1,
                flow=25.0,
                source_cell=500,
                mouth_cell=4000,
                cell_count=15,
            ),
        ]


@app.get("/maps/{map_id}/cultures", response_model=List[CultureInfo])
async def get_map_cultures(map_id: str):
    """Get culture information for a map."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        # For now, return mock culture data
        # In a full implementation, this would load the actual culture data from the database

        return [
            CultureInfo(
                id=1,
                name="Northmen",
                color="#9e2a2b",
                type="Highland",
                area_km2=15000.0,
                population=250000,
                expansionism=1.2,
                center_cell=1500,
            ),
            CultureInfo(
                id=2,
                name="Islanders",
                color="#4682b4",
                type="Naval",
                area_km2=8000.0,
                population=180000,
                expansionism=1.5,
                center_cell=2800,
            ),
            CultureInfo(
                id=3,
                name="Desert Riders",
                color="#daa520",
                type="Nomadic",
                area_km2=20000.0,
                population=120000,
                expansionism=0.8,
                center_cell=3200,
            ),
            CultureInfo(
                id=4,
                name="Forest Folk",
                color="#0f8040",
                type="Generic",
                area_km2=12000.0,
                population=200000,
                expansionism=1.0,
                center_cell=1800,
            ),
            CultureInfo(
                id=5,
                name="River People",
                color="#6ba9cb",
                type="River",
                area_km2=9000.0,
                population=160000,
                expansionism=1.1,
                center_cell=2200,
            ),
        ]


@app.get("/maps/{map_id}/religions", response_model=List[ReligionInfo])
async def get_map_religions(map_id: str):
    """Get religion information for a map."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        # For now, return mock religion data
        # In a full implementation, this would load the actual religion data from the database

        return [
            ReligionInfo(
                id=1,
                name="The Old Gods",
                color="#8b4513",
                type="Folk",
                form="Shamanism",
                deity=None,
                expansion="culture",
                expansionism=0.5,
                area_km2=15000.0,
                rural_population=200000.0,
                urban_population=50000.0,
            ),
            ReligionInfo(
                id=2,
                name="Church of the Sun",
                color="#ffd700",
                type="Organized",
                form="Monotheism",
                deity="Solaris",
                expansion="global",
                expansionism=2.0,
                area_km2=25000.0,
                rural_population=300000.0,
                urban_population=120000.0,
            ),
            ReligionInfo(
                id=3,
                name="Order of the Deep",
                color="#1a4b5c",
                type="Organized",
                form="Polytheism",
                deity="Thalassa",
                expansion="state",
                expansionism=1.5,
                area_km2=12000.0,
                rural_population=150000.0,
                urban_population=80000.0,
            ),
            ReligionInfo(
                id=4,
                name="Wind Walkers",
                color="#87ceeb",
                type="Folk",
                form="Animism",
                deity=None,
                expansion="culture",
                expansionism=0.8,
                area_km2=18000.0,
                rural_population=180000.0,
                urban_population=40000.0,
            ),
        ]


@app.get("/maps/{map_id}/settlements", response_model=List[SettlementInfo])
async def get_map_settlements(map_id: str):
    """Get settlement information for a map."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()

        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")

        # For now, return mock settlement data
        # In a full implementation, this would load the actual settlement data from the database

        return [
            SettlementInfo(
                id=1,
                name="Ironhold",
                type="capital",
                population=45000,
                is_capital=True,
                is_port=False,
                culture_name="Northmen",
                state_name="Northern Kingdom",
                religion_name="The Old Gods",
            ),
            SettlementInfo(
                id=2,
                name="Seahaven",
                type="city",
                population=28000,
                is_capital=False,
                is_port=True,
                culture_name="Islanders",
                state_name="Maritime Republic",
                religion_name="Order of the Deep",
            ),
            SettlementInfo(
                id=3,
                name="Goldspire",
                type="capital",
                population=38000,
                is_capital=True,
                is_port=False,
                culture_name="Desert Riders",
                state_name="Desert Emirates",
                religion_name="Church of the Sun",
            ),
            SettlementInfo(
                id=4,
                name="Greenwood",
                type="town",
                population=12000,
                is_capital=False,
                is_port=False,
                culture_name="Forest Folk",
                state_name="Woodland Alliance",
                religion_name="The Old Gods",
            ),
            SettlementInfo(
                id=5,
                name="Rivermouth",
                type="city",
                population=22000,
                is_capital=False,
                is_port=True,
                culture_name="River People",
                state_name="River Confederacy",
                religion_name="Wind Walkers",
            ),
        ]


# Background task functions
async def run_map_generation(job_id: str, request: MapGenerationRequest) -> None:
    """
    Background task to generate a map.

    This is a simplified implementation - the full version would include
    all the generation stages (heightmap, climate, rivers, biomes, etc.)
    """
    # request = MapGenerationRequest(**request_data)  # Reconstruct request object
    logger.info("Starting map generation", job_id=job_id)

    try:
        # Load job once and reuse its data
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.status = "running"
            job.started_at = datetime.utcnow()
            session.commit()
            seed = job.seed

        # Stage 1: Generate Voronoi graph
        logger.info("Generating Voronoi graph", job_id=job_id)
        config = GridConfig(
            width=request.width,
            height=request.height,
            cells_desired=request.cells_desired,
        )
        voronoi_graph = generate_voronoi_graph(config, seed)

        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 10
            session.commit()

        # Stage 2: Heightmap generation

        logger.info("Generating heightmap", job_id=job_id)
        heightmap_config = HeightmapConfig(
            width=request.width,
            height=request.height,
            cells_x=voronoi_graph.cells_x,
            cells_y=voronoi_graph.cells_y,
            cells_desired=request.cells_desired,
            spacing=voronoi_graph.spacing,
        )
        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
        heights = heightmap_gen.from_template(request.template_name, seed)
        voronoi_graph.heights = heights

        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 30
            session.commit()

        # Stage 3: Coastlines

        logger.info("Marking up coastlines", job_id=job_id)
        features = Features(voronoi_graph, seed=seed)
        features.markup_grid()

        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 32
            session.commit()

        # Stage 4: ReGraph
        logger.info("Performing reGraph coastal resampling", job_id=job_id)
        packed_graph = regraph(voronoi_graph)
        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 35
            session.commit()

        # Stage 5: Pack data
        logger.info("Packing reGraphed data", job_id=job_id)
        packed_heights = packed_graph.heights

        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 40
            session.commit()

        # Stage 6: Climate
        logger.info("Generating climate", job_id=job_id)
        climate = Climate(
            packed_graph,
            options=ClimateOptions(),
            map_coords=MapCoordinates(lat_n=90, lat_s=-90),
        )
        climate.calculate_temperatures()
        climate.generate_precipitation()
        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 60
            session.commit()

        # Stage 7: Hydrology
        logger.info("Generating hydrology", job_id=job_id)
        hydrology = Hydrology(packed_graph, options=HydrologyOptions())
        hydrology.run_full_simulation()
        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 70
            session.commit()

        # Stage 8: Biomes
        logger.info("Generating biomes", job_id=job_id)
        biomes = BiomeClassifier(packed_graph, options=BiomeOptions())
        biomes.run_full_classification()
        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 80
            session.commit()

        # Final: Save map
        logger.info("Saving map", job_id=job_id)
        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            map_name = request.map_name or f"Map {job.seed}"

            map_obj = Map(
                name=map_name,
                seed=job.seed,
                width=job.width,
                height=job.height,
                cells_count=len(packed_graph.points),
                generation_time_seconds=1.0,  # TODO: track actual time
            )
            session.add(map_obj)
            session.flush()
            map_id = map_obj.id
            job.map_id = map_obj.id

            job.status = "completed"
            job.progress_percent = 100
            job.completed_at = datetime.utcnow()
            session.commit()

        logger.info("Map generation completed", job_id=job_id, map_id=str(map_id))

    except Exception as e:
        logger.error("Map generation failed", job_id=job_id, error=str(e))

        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.commit()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
