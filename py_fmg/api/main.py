"""FastAPI main application."""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from typing import Optional
import structlog
import uuid
from datetime import datetime

from ..config import settings
from ..db.connection import db
from ..db.models import GenerationJob, Map
from ..core.voronoi_graph import GridConfig, generate_voronoi_graph
from ..core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from ..core.cell_packing import regraph
from ..core.features import Features
from ..core.climate import Climate, ClimateOptions, MapCoordinates
from ..core.hydrology import Hydrology, HydrologyOptions
from ..core.biomes import BiomeClassifier, BiomeOptions

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

# Initialize FastAPI app
app = FastAPI(
    title="Fantasy Map Generator API",
    description="Python port of Azgaar's Fantasy Map Generator",
    version="0.1.0"
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
    
    seed: Optional[str] = Field(None, description="Random seed for reproducible generation (deprecated, use grid_seed)")
    grid_seed: Optional[str] = Field(None, description="Random seed for grid/Voronoi generation")
    map_seed: Optional[str] = Field(None, description="Random seed for heightmap generation")
    width: float = Field(800, ge=100, le=2000, description="Map width")
    height: float = Field(600, ge=100, le=2000, description="Map height")
    cells_desired: int = Field(10000, ge=1000, le=50000, description="Target number of cells")
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
    seed: str  # Legacy field for backwards compatibility
    grid_seed: str  # Seed used for Voronoi grid generation
    map_seed: str   # Seed used for heightmap generation
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


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Starting Fantasy Map Generator API")
    db.initialize()
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Fantasy Map Generator API")


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fantasy Map Generator API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        with db.get_session() as session:
            session.execute("SELECT 1")
        
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/maps/generate", response_model=JobResponse)
async def generate_map(request: MapGenerationRequest, background_tasks: BackgroundTasks):
    """
    Start map generation job.
    
    Returns immediately with job ID. Use /jobs/{job_id} to check status.
    """
    logger.info("Map generation requested", request=request.dict())
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Handle seed logic: support both legacy single seed and new dual seed system
    if request.grid_seed or request.map_seed:
        # New dual seed system
        grid_seed = request.grid_seed or request.map_seed or str(uuid.uuid4())[:8]
        map_seed = request.map_seed or request.grid_seed or str(uuid.uuid4())[:8]
    else:
        # Legacy single seed or fallback
        legacy_seed = request.seed or str(uuid.uuid4())[:8]
        grid_seed = legacy_seed
        map_seed = legacy_seed
    
    # Create job record
    with db.get_session() as session:
        job = GenerationJob(
            id=job_id,
            seed=request.seed or grid_seed,  # Keep legacy field populated
            grid_seed=grid_seed,
            map_seed=map_seed,
            width=request.width,
            height=request.height,
            cells_desired=request.cells_desired,
            template_name=request.template_name,
            status="pending"
        )
        session.add(job)
        session.commit()
    
    # Start background generation
    background_tasks.add_task(
        run_map_generation,
        job_id,
        request
    )
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        progress_percent=0,
        message="Map generation job started"
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
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
            error_message=job.error_message
        )


@app.get("/maps", response_model=list[MapSummary])
async def list_maps():
    """List all generated maps."""
    with db.get_session() as session:
        maps = session.query(Map).order_by(Map.created_at.desc()).all()
        
        return [
            MapSummary(
                id=str(map.id),
                name=map.name,
                seed=map.seed,
                grid_seed=map.grid_seed,
                map_seed=map.map_seed,
                width=map.width,
                height=map.height,
                cells_count=map.cells_count,
                created_at=map.created_at,
                generation_time_seconds=map.generation_time_seconds
            )
            for map in maps
        ]


@app.get("/maps/{map_id}", response_model=MapSummary)
async def get_map(map_id: str):
    """Get map details."""
    with db.get_session() as session:
        map_obj = session.query(Map).filter(Map.id == map_id).first()
        
        if not map_obj:
            raise HTTPException(status_code=404, detail="Map not found")
        
        return MapSummary(
            id=str(map_obj.id),
            name=map_obj.name,
            seed=map_obj.seed,
            grid_seed=map_obj.grid_seed,
            map_seed=map_obj.map_seed,
            width=map_obj.width,
            height=map_obj.height,
            cells_count=map_obj.cells_count,
            created_at=map_obj.created_at,
            generation_time_seconds=map_obj.generation_time_seconds
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
            BiomeStatistics(biome_name="Ocean", cell_count=3000, percentage=30.0, avg_temperature=15.0, avg_precipitation=100.0),
            BiomeStatistics(biome_name="Temperate Forest", cell_count=2500, percentage=25.0, avg_temperature=12.0, avg_precipitation=120.0),
            BiomeStatistics(biome_name="Grassland", cell_count=2000, percentage=20.0, avg_temperature=18.0, avg_precipitation=80.0),
            BiomeStatistics(biome_name="Desert", cell_count=1500, percentage=15.0, avg_temperature=25.0, avg_precipitation=20.0),
            BiomeStatistics(biome_name="Mountains", cell_count=1000, percentage=10.0, avg_temperature=5.0, avg_precipitation=150.0),
        ]
        
        mock_rivers = [
            RiverInfo(id=1, length=250.5, flow=150.0, source_cell=100, mouth_cell=5000, cell_count=45),
            RiverInfo(id=2, length=180.2, flow=80.0, source_cell=200, mouth_cell=4800, cell_count=32),
            RiverInfo(id=3, length=120.8, flow=45.0, source_cell=300, mouth_cell=4500, cell_count=22),
        ]
        
        return MapStatistics(
            map_id=str(map_obj.id),
            total_cells=map_obj.cells_count,
            land_cells=int(map_obj.cells_count * 0.7),  # Mock: 70% land
            water_cells=int(map_obj.cells_count * 0.3),  # Mock: 30% water
            rivers_count=15,  # Mock
            lakes_count=5,    # Mock
            biome_distribution=mock_biome_stats,
            major_rivers=mock_rivers,
            temperature_range=(-10.0, 35.0),  # Mock temperature range
            precipitation_range=(10.0, 200.0)  # Mock precipitation range
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
            height=45  # Mock height
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
            BiomeStatistics(biome_name="Ocean", cell_count=3000, percentage=30.0, avg_temperature=15.0, avg_precipitation=100.0),
            BiomeStatistics(biome_name="Temperate Deciduous Forest", cell_count=2500, percentage=25.0, avg_temperature=12.0, avg_precipitation=120.0),
            BiomeStatistics(biome_name="Temperate Grassland", cell_count=2000, percentage=20.0, avg_temperature=18.0, avg_precipitation=80.0),
            BiomeStatistics(biome_name="Desert", cell_count=1500, percentage=15.0, avg_temperature=25.0, avg_precipitation=20.0),
            BiomeStatistics(biome_name="Alpine", cell_count=1000, percentage=10.0, avg_temperature=5.0, avg_precipitation=150.0),
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
            RiverInfo(id=1, length=250.5, flow=150.0, source_cell=100, mouth_cell=5000, cell_count=45),
            RiverInfo(id=2, length=180.2, flow=80.0, source_cell=200, mouth_cell=4800, cell_count=32),
            RiverInfo(id=3, length=120.8, flow=45.0, source_cell=300, mouth_cell=4500, cell_count=22),
            RiverInfo(id=4, length=95.3, flow=35.0, source_cell=400, mouth_cell=4200, cell_count=18),
            RiverInfo(id=5, length=75.1, flow=25.0, source_cell=500, mouth_cell=4000, cell_count=15),
        ]


# Background task functions
async def run_map_generation(job_id: str, request: dict):
    """
    Background task to generate a map.
    """
    # request = MapGenerationRequest(**request_data)  # Reconstruct request object
    logger.info("Starting map generation", job_id=job_id)

    try:
        # Load job once and reuse its data
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.status = "running"
            job.started_at = datetime.utcnow()
            session.commit()
            grid_seed = job.grid_seed
            map_seed = job.map_seed

        # Stage 1: Generate Voronoi graph
        logger.info("Generating Voronoi graph", job_id=job_id)
        config = GridConfig(width=request.width, height=request.height, cells_desired=request.cells_desired)
        voronoi_graph = generate_voronoi_graph(config, grid_seed)

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
            cells_desired=request.cells_desired
        )
        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
        heights = heightmap_gen.from_template(request.template_name, map_seed)
        voronoi_graph.heights = heights

        with db.get_session() as session:
            job = session.query(GenerationJob).get(job_id)
            job.progress_percent = 30
            session.commit()

        # Stage 3: Coastlines
        logger.info("Marking up coastlines", job_id=job_id)
        features = Features(voronoi_graph)
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
        climate = Climate(packed_graph, options=ClimateOptions(), map_coords=MapCoordinates(lat_n=90, lat_s=-90))
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
            map_name = request.map_name or f"Map {job.grid_seed}"

            map_obj = Map(
                name=map_name,
                seed=job.seed,
                grid_seed=job.grid_seed,
                map_seed=job.map_seed,
                width=job.width,
                height=job.height,
                cells_count=len(packed_graph.points),
                generation_time_seconds=1.0  # TODO: track actual time
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