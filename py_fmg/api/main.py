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
from ..core.voronoi_graph import GridConfig, generate_voronoi_graph

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
    
    seed: Optional[str] = Field(None, description="Random seed for reproducible generation")
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
    seed: str
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
    
    # Create job record
    with db.get_session() as session:
        job = GenerationJob(
            id=job_id,
            seed=request.seed or str(uuid.uuid4())[:8],
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
        
        # Use seed from request or job
        seed = request.seed
        if not seed:
            with db.get_session() as session:
                job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
                seed = job.seed
        
        voronoi_graph = generate_voronoi_graph(config, seed)
        
        # Update progress
        with db.get_session() as session:
            job = session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            job.progress_percent = 10
            session.commit()
        
        # TODO: Add remaining generation stages:
        # Stage 2: Generate heightmap (20% progress)
        # Stage 3: Generate climate (40% progress)  
        # Stage 4: Generate rivers (60% progress)
        # Stage 5: Generate biomes (70% progress)
        # Stage 6: Generate settlements (80% progress)
        # Stage 7: Generate states (90% progress)
        # Stage 8: Save to database (100% progress)
        
        # For now, create a basic map record
        map_name = request.map_name or f"Map {seed}"
        
        with db.get_session() as session:
            map_obj = Map(
                name=map_name,
                seed=seed,
                width=request.width,
                height=request.height,
                cells_count=len(voronoi_graph.points),
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