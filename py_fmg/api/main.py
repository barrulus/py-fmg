"""FastAPI main application."""

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
from ..core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from ..core.hydrology import Hydrology
from ..core.name_generator import NameGenerator
from ..core.settlements import Settlements
from ..core.voronoi_graph import GridConfig, generate_voronoi_graph
from ..db.connection import db
from ..db.models import GenerationJob, Map

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

    seed: Optional[str] = Field(
        None, description="Random seed for reproducible generation (deprecated, use grid_seed)"
    )
    grid_seed: Optional[str] = Field(
        None, description="Random seed for grid/Voronoi generation"
    )
    map_seed: Optional[str] = Field(
        None, description="Random seed for heightmap generation"
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
    seed: str  # Legacy field for backwards compatibility
    grid_seed: str  # Seed used for Voronoi grid generation
    map_seed: str   # Seed used for heightmap generation
    width: float
    height: float
    cells_count: int
    created_at: datetime
    generation_time_seconds: Optional[float]


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
        "status": "running"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
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
async def generate_map(request: MapGenerationRequest, background_tasks: BackgroundTasks) -> JobResponse:
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
            error_message=job.error_message
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
            grid_seed=map_obj.grid_seed,
            map_seed=map_obj.map_seed,
            width=map_obj.width,
            height=map_obj.height,
            cells_count=map_obj.cells_count,
            created_at=map_obj.created_at,
            generation_time_seconds=map_obj.generation_time_seconds
        )


# Background task functions
async def run_map_generation(job_id: str, request: MapGenerationRequest) -> None:
    """
    Background task to generate a map.
    
    This is a simplified implementation - the full version would include
    all the generation stages (heightmap, climate, rivers, biomes, etc.)
    """
    logger.info("Starting map generation", job_id=job_id)

    try:
        # Update job status
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.status = "running"
            job.started_at = datetime.utcnow()
            session.commit()

        # Stage 1: Generate Voronoi graph (10% progress)
        logger.info("Generating Voronoi graph", job_id=job_id)
        config = GridConfig(
            width=request.width,
            height=request.height,
            cells_desired=request.cells_desired
        )

        # Get grid seed from job (should use grid_seed for Voronoi generation)
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            grid_seed = job.grid_seed

        voronoi_graph = generate_voronoi_graph(config, grid_seed)

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 10
            session.commit()

        # Stage 2: Generate heightmap (30% progress)
        logger.info("Generating heightmap", job_id=job_id)
        heightmap_config = HeightmapConfig(
            width=int(request.width),
            height=int(request.height),
            cells_x=voronoi_graph.cells_x,
            cells_y=voronoi_graph.cells_y,
            cells_desired=request.cells_desired
        )

        # Get map seed from job (should use map_seed for heightmap generation)
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            map_seed = job.map_seed

        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
        heights = heightmap_gen.from_template(request.template_name, map_seed)

        # Assign heights to the graph
        voronoi_graph.heights = heights

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 30
            session.commit()

        # Stage 3: Mark up coastlines with Features (32% progress)
        logger.info("Marking up coastlines", job_id=job_id)
        features = Features(voronoi_graph)
        features.markup_grid()

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 32
            session.commit()

        # Stage 4: Perform reGraph coastal resampling (35% progress)
        logger.info("Performing reGraph coastal resampling", job_id=job_id)
        packed_graph = regraph(voronoi_graph)

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 35
            session.commit()

        # Stage 5: Pack the reGraphed data (40% progress)
        logger.info("Packing reGraphed data", job_id=job_id)
        # The packed_graph already contains the new Voronoi graph and heights
        packed_heights = packed_graph.heights

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 40
            session.commit()

        # Stage 6: Generate climate (60% progress)
        logger.info("Generating climate", job_id=job_id)
        climate = Climate(packed_graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 60
            session.commit()

        # Stage 7: Generate rivers (70% progress)
        logger.info("Generating rivers", job_id=job_id)
        hydrology = Hydrology(packed_graph, features, climate)
        rivers = hydrology.generate_rivers()

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 70
            session.commit()

        # Stage 8: Generate cultures (75% progress)
        logger.info("Generating cultures", job_id=job_id)
        culture_generator = CultureGenerator(packed_graph, features)
        cultures_dict, cell_cultures, cell_population, cell_suitability = culture_generator.generate()

        # Store population data on the graph for settlements to use
        packed_graph.cell_population = cell_population
        packed_graph.cell_suitability = cell_suitability

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 75
            session.commit()

        # Stage 9: Generate biomes (80% progress)
        logger.info("Generating biomes", job_id=job_id)
        biome_classifier = BiomeClassifier()

        # Prepare river data for biome classification
        has_river = np.zeros(len(packed_graph.points), dtype=bool)
        river_flux = np.zeros(len(packed_graph.points), dtype=float)

        for river_id, river_data in rivers.items():
            for cell_id in river_data.cells:
                if cell_id < len(has_river):  # Safety check
                    has_river[cell_id] = True
                    river_flux[cell_id] = max(river_flux[cell_id], river_data.discharge)

        # Classify biomes using climate and terrain data
        cell_biomes = biome_classifier.classify_biomes(
            temperatures=climate.temperatures,
            precipitation=climate.precipitation,
            heights=packed_graph.heights,
            river_flux=river_flux,
            has_river=has_river
        )

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 80
            session.commit()

        # Stage 10: Generate religions (85% progress)
        logger.info("Generating religions", job_id=job_id)
        from ..core.religions import ReligionGenerator
        
        # Create name generator for both religions and settlements
        name_generator = NameGenerator()
        
        religion_generator = ReligionGenerator(
            packed_graph,
            cultures_dict,
            cell_cultures,
            {},  # settlements_dict (empty at this stage)
            {},  # states_dict (would be generated later in full implementation)
            name_generator=name_generator  # Pass name generator for culture-based deity names
        )
        religions_dict, cell_religions = religion_generator.generate()
        
        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 85
            session.commit()

        # Stage 11: Generate settlements (90% progress)
        logger.info("Generating settlements", job_id=job_id)

        # Create cultures wrapper that matches expected interface
        class CulturesWrapper:
            def __init__(self, cultures_dict: Dict, cell_cultures: np.ndarray) -> None:
                self.cultures = cultures_dict
                self.cell_cultures = cell_cultures

        class BiomeWrapper:
            def __init__(
                self, cell_biomes: np.ndarray, classifier: BiomeClassifier
            ) -> None:
                self.cell_biomes = cell_biomes
                self.classifier = classifier

            def get_biome_properties(self, biome_id: int) -> Dict:
                return self.classifier.get_biome_properties(biome_id)

        cultures_wrapper = CulturesWrapper(cultures_dict, cell_cultures)
        biome_wrapper = BiomeWrapper(cell_biomes, biome_classifier)
        settlements = Settlements(
            packed_graph, 
            features, 
            cultures_wrapper, 
            biome_wrapper, 
            name_generator,
            cell_religions=cell_religions
        )
        settlements.generate()
        
        # Assign temples based on religion system
        religion_generator.assign_temples_to_settlements(settlements.settlements)

        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 90
            session.commit()

        # Stage 12: Save to database (100% progress)

        # For now, create a basic map record
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            map_name = request.map_name or f"Map {job.grid_seed}"

            map_obj = Map(
                name=map_name,
                seed=job.seed,  # Legacy field
                grid_seed=job.grid_seed,
                map_seed=job.map_seed,
                width=request.width,
                height=request.height,
                cells_count=len(packed_graph.points),  # Use packed graph cell count
                generation_time_seconds=1.0  # Placeholder
            )
            session.add(map_obj)
            session.flush()  # Get the ID

            # Update job
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.map_id = map_obj.id
            job.status = "completed"
            job.progress_percent = 100
            job.completed_at = datetime.utcnow()
            session.commit()

        logger.info("Map generation completed", job_id=job_id, map_id=str(map_obj.id))

    except Exception as e:
        logger.error("Map generation failed", job_id=job_id, error=str(e))

        # Update job with error
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.commit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

