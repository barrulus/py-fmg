"""FastAPI main application."""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import structlog
import uuid
from datetime import datetime

from ..config import settings
from ..db.connection import db
from ..db.models import GenerationJob, Map
from ..core.voronoi_graph import GridConfig, generate_voronoi_graph, pack_graph
from ..core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from ..core.regraph import regraph

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


# Background task functions
async def run_map_generation(job_id: str, request: MapGenerationRequest):
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
        
        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 30
            session.commit()
        
        # Stage 3: Perform reGraph coastal resampling (35% progress)
        logger.info("Performing reGraph coastal resampling", job_id=job_id)
        regraph_result = regraph(voronoi_graph, heights, config, grid_seed)
        
        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 35
            session.commit()
        
        # Stage 4: Pack the reGraphed data (40% progress)
        logger.info("Packing reGraphed data", job_id=job_id)
        # The reGraph result already contains the new Voronoi graph and heights
        packed_graph = regraph_result.voronoi_graph
        packed_heights = regraph_result.heights
        
        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 40
            session.commit()
        
        # TODO: Add remaining generation stages:
        # Stage 4: Generate climate (60% progress)  
        # Stage 5: Generate rivers (70% progress)
        # Stage 6: Generate biomes (80% progress)
        # Stage 7: Generate settlements (90% progress)
        # Stage 8: Generate states (95% progress)
        # Stage 9: Save to database (100% progress)
        
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