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

# Configure logging
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
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.status = "running"
            job.started_at = datetime.utcnow()
            session.commit()

        # Stage 1: Generate Voronoi graph (10% progress)
        logger.info("Generating Voronoi graph", job_id=job_id)

        # Get job details and seed
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            seed = job.seed

        config = GridConfig(
            width=request.width,
            height=request.height,
            cells_desired=request.cells_desired,
        )

        voronoi_graph = generate_voronoi_graph(config, seed)

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 10
            session.commit()

        # Stage 2: Generate heightmap (30% progress)
        logger.info("Generating heightmap", job_id=job_id)
        heightmap_config = HeightmapConfig(
            width=int(request.width),
            height=int(request.height),
            cells_x=voronoi_graph.cells_x,
            cells_y=voronoi_graph.cells_y,
            cells_desired=request.cells_desired,
            spacing=voronoi_graph.spacing,
        )

        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
        heights = heightmap_gen.from_template(request.template_name, seed)

        # Assign heights to the graph
        voronoi_graph.heights = heights

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 30
            session.commit()

        # Stage 3: Mark up coastlines with Features (32% progress)
        logger.info("Marking up coastlines", job_id=job_id)
        features = Features(voronoi_graph, seed=seed)
        features.markup_grid()

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 32
            session.commit()

        # Stage 4: Perform reGraph coastal resampling (35% progress)
        logger.info("Performing reGraph coastal resampling", job_id=job_id)
        packed_graph = regraph(voronoi_graph)

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 35
            session.commit()

        # Stage 5: Pack the reGraphed data (40% progress)
        logger.info("Packing reGraphed data", job_id=job_id)
        # The packed_graph already contains the new Voronoi graph and heights
        packed_heights = packed_graph.heights

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 40
            session.commit()

        # Create Map record early so we can reference it for spatial data
        logger.info("Creating map record", job_id=job_id)

        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            map_name = request.map_name or f"Map {seed}"

            map_obj = Map(
                name=map_name,
                seed=seed,
                width=request.width,
                height=request.height,
                cells_count=len(packed_graph.points),  # Use packed graph cell count
                generation_time_seconds=0.0,  # Will update at end
            )
            session.add(map_obj)
            session.flush()  # Get the ID

            # Store map_id for use in subsequent exports
            map_id = map_obj.id

            # Update job with map_id
            job.map_id = map_id
            session.commit()

        # Export Voronoi cells to database (right after Stage 5)
        logger.info("Exporting Voronoi cells to database", job_id=job_id)

        # We need to create polygon geometries from the Voronoi cells
        # For now, let's create simple point-based cells and improve later
        from shapely.geometry import Point
        from geoalchemy2.shape import from_shape

        with db.get_session() as session:
            for i, (point, height) in enumerate(
                zip(packed_graph.points, packed_graph.heights)
            ):
                # Create a simple circular polygon around each point (temporary solution)
                # In a full implementation, we'd use the actual Voronoi cell boundaries
                x, y = point[0], point[1]

                # Create a small polygon around the point (radius based on map size)
                radius = (
                    min(request.width, request.height) / len(packed_graph.points) * 2
                )
                circle = Point(x, y).buffer(radius)

                voronoi_cell = VoronoiCell(
                    map_id=map_id,
                    cell_index=i,
                    geometry=from_shape(circle, srid=4326),
                    height=int(height),
                    is_land=height >= 20,
                    is_coastal=False,  # Will be determined later
                    center_x=float(x),
                    center_y=float(y),
                    area=float(circle.area),
                )
                session.add(voronoi_cell)

            session.commit()
            logger.info(
                f"Exported {len(packed_graph.points)} Voronoi cells to database"
            )

        # Stage 6: Generate climate (60% progress)
        logger.info("Generating climate", job_id=job_id)
        # Note: Climate uses ClimateOptions with sensible defaults
        # For custom climate parameters, pass ClimateOptions() with desired settings
        climate = Climate(packed_graph)
        climate.calculate_temperatures()
        climate.generate_precipitation()

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 60
            session.commit()

        # Export climate data to database (right after Stage 6)
        logger.info("Exporting climate data to database", job_id=job_id)

        with db.get_session() as session:
            for i, (point, temp, precip) in enumerate(
                zip(packed_graph.points, climate.temperatures, climate.precipitation)
            ):
                x, y = point[0], point[1]

                # Calculate latitude for reference (assuming map spans -90 to +90 degrees proportionally)
                latitude = 90.0 - (y / request.height) * 180.0

                # Get altitude from height
                altitude = packed_graph.heights[i]

                climate_data = ClimateData(
                    map_id=map_id,
                    cell_index=i,
                    temperature_c=int(temp),
                    precipitation_mm=int(precip),
                    latitude=float(latitude),
                    altitude_m=int(altitude),
                    moisture_index=0,  # Will be calculated later if needed
                    temperature_index=0,  # Will be calculated later if needed
                )
                session.add(climate_data)

            session.commit()
            logger.info(
                f"Exported climate data for {len(climate.temperatures)} cells to database"
            )

        # Stage 7: Generate rivers (70% progress)
        logger.info("Generating rivers", job_id=job_id)
        hydrology = Hydrology(packed_graph, features, climate)
        rivers = hydrology.generate_rivers()

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 70
            session.commit()

        # Export rivers to database (right after Stage 7)
        logger.info("Exporting rivers to database", job_id=job_id)

        from shapely.geometry import LineString
        from geoalchemy2.shape import from_shape

        with db.get_session() as session:
            for river_id, river_data in rivers.items():
                if (
                    len(river_data.cells) >= 2
                ):  # Need at least 2 points for a linestring
                    # Create linestring from river cells
                    river_points = []
                    for cell_idx in river_data.cells:
                        if cell_idx < len(packed_graph.points):
                            point = packed_graph.points[cell_idx]
                            river_points.append((point[0], point[1]))

                    if len(river_points) >= 2:
                        linestring = LineString(river_points)

                        # Generate a simple river name
                        river_name = f"River {river_id}"

                        river_record = River(
                            map_id=map_id,
                            river_index=river_id,
                            name=river_name,
                            geometry=from_shape(linestring, srid=4326),
                            length_km=float(linestring.length),  # Approximate length
                            discharge_m3s=float(river_data.discharge),
                            average_width_m=5.0,  # Default width
                            is_main_stem=True,  # Simplification for now
                            source_elevation_m=0.0,  # Will be calculated later
                            mouth_elevation_m=0.0,  # Will be calculated later
                        )
                        session.add(river_record)

            session.commit()
            logger.info(f"Exported {len(rivers)} rivers to database")

        # Stage 8: Generate cultures (75% progress)
        logger.info("Generating cultures", job_id=job_id)
        # Create BiomeClassifier for culture generation
        biome_classifier = BiomeClassifier()
        culture_generator = CultureGenerator(packed_graph, features, biome_classifier)
        (
            cultures_dict,
            cell_cultures,
            cell_population,
            cell_suitability,
        ) = culture_generator.generate()

        # Store population data on the graph for settlements to use
        packed_graph.cell_population = cell_population
        packed_graph.cell_suitability = cell_suitability

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 75
            session.commit()

        # Export cultures to database (right after Stage 8)
        logger.info("Exporting cultures to database", job_id=job_id)

        from shapely.geometry import Polygon, MultiPolygon
        from geoalchemy2.shape import from_shape
        import numpy as np

        with db.get_session() as session:
            # Export culture definitions
            for culture_id, culture_data in cultures_dict.items():
                culture_record = Culture(
                    map_id=map_id,
                    culture_index=culture_id,
                    name=culture_data.name,
                    color=culture_data.color,
                    type=culture_data.type,
                    expansionism=float(culture_data.expansionism),
                    center_cell_index=culture_data.center,
                    name_base=culture_data.name_base,
                    area_km2=0.0,  # Will calculate from cells
                    population=0,  # Will calculate from cells
                    cells_count=0,  # Will calculate from cells
                )
                session.add(culture_record)
                session.flush()  # Get the culture ID

                # Export cell assignments for this culture
                culture_cells = np.where(cell_cultures == culture_id)[0]
                total_population = 0

                for cell_idx in culture_cells:
                    if cell_idx < len(packed_graph.points):
                        cell_culture = CellCulture(
                            map_id=map_id,
                            cell_index=int(cell_idx),
                            culture_id=culture_record.id,
                            population=float(cell_population[cell_idx])
                            if cell_idx < len(cell_population)
                            else 0.0,
                            suitability=int(cell_suitability[cell_idx])
                            if cell_idx < len(cell_suitability)
                            else 0,
                        )
                        session.add(cell_culture)
                        total_population += float(cell_culture.population)

                # Update culture statistics
                culture_record.cells_count = len(culture_cells)
                culture_record.population = int(total_population)

                # Create a simple polygon geometry from culture cells (convex hull)
                if len(culture_cells) >= 3:
                    culture_points = [
                        packed_graph.points[i]
                        for i in culture_cells
                        if i < len(packed_graph.points)
                    ]
                    if len(culture_points) >= 3:
                        try:
                            from scipy.spatial import ConvexHull

                            hull = ConvexHull(culture_points)
                            hull_points = [culture_points[i] for i in hull.vertices]
                            polygon = Polygon(hull_points)
                            culture_record.geometry = from_shape(polygon, srid=4326)
                            culture_record.area_km2 = float(polygon.area)
                        except:
                            # Fallback: create a simple polygon around center
                            center_point = packed_graph.points[culture_data.center]
                            radius = 50.0  # Default radius
                            from shapely.geometry import Point

                            circle = Point(center_point[0], center_point[1]).buffer(
                                radius
                            )
                            culture_record.geometry = from_shape(circle, srid=4326)
                            culture_record.area_km2 = float(circle.area)

            session.commit()
            logger.info(
                f"Exported {len(cultures_dict)} cultures and {np.sum(cell_cultures >= 0)} culture-cell assignments to database"
            )

        # Stage 9: Generate biomes (80% progress)
        logger.info("Generating biomes", job_id=job_id)
        # Use existing biome_classifier from cultures stage

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
            has_river=has_river,
        )

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 80
            session.commit()

        # Export biomes to database (right after Stage 9)
        logger.info("Exporting biomes to database", job_id=job_id)

        with db.get_session() as session:
            # Group cells by biome type
            biome_groups = {}
            for cell_idx, biome_id in enumerate(cell_biomes):
                if biome_id not in biome_groups:
                    biome_groups[biome_id] = []
                biome_groups[biome_id].append(cell_idx)

            # Export each biome region
            for biome_id, cells in biome_groups.items():
                if len(cells) > 0:
                    biome_props = biome_classifier.get_biome_properties(biome_id)
                    biome_name = biome_props.get("name", f"Biome {biome_id}")

                    # Calculate average temperature and precipitation for this biome
                    avg_temp = np.mean(
                        [
                            climate.temperatures[i]
                            for i in cells
                            if i < len(climate.temperatures)
                        ]
                    )
                    avg_precip = np.mean(
                        [
                            climate.precipitation[i]
                            for i in cells
                            if i < len(climate.precipitation)
                        ]
                    )

                    # Create a simple polygon geometry from biome cells (convex hull)
                    biome_points = [
                        packed_graph.points[i]
                        for i in cells
                        if i < len(packed_graph.points)
                    ]
                    polygon_geom = None
                    area = 0.0

                    if len(biome_points) >= 3:
                        try:
                            from scipy.spatial import ConvexHull

                            hull = ConvexHull(biome_points)
                            hull_points = [biome_points[i] for i in hull.vertices]
                            polygon = Polygon(hull_points)
                            polygon_geom = from_shape(polygon, srid=4326)
                            area = float(polygon.area)
                        except:
                            # Fallback: create circle around first point
                            center = biome_points[0]
                            from shapely.geometry import Point

                            circle = Point(center[0], center[1]).buffer(25.0)
                            polygon_geom = from_shape(circle, srid=4326)
                            area = float(circle.area)

                    biome_region = BiomeRegion(
                        map_id=map_id,
                        biome_type=biome_name,
                        biome_index=int(biome_id),  # Convert numpy types to int
                        biome_classification=biome_props.get(
                            "classification", "Unknown"
                        ),
                        geometry=polygon_geom,
                        area_km2=area,
                        habitability_score=int(
                            biome_props.get("habitability", 50)
                        ),  # Convert to int
                        movement_cost=int(
                            biome_props.get("movement_cost", 1)
                        ),  # Convert to int
                        avg_temperature_c=float(avg_temp),
                        avg_precipitation_mm=float(avg_precip),
                    )
                    session.add(biome_region)

            session.commit()
            logger.info(f"Exported {len(biome_groups)} biome regions to database")

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
            name_generator=name_generator,  # Pass name generator for culture-based deity names
        )
        religions_dict, cell_religions = religion_generator.generate()

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 85
            session.commit()

        # Export religions to database (right after Stage 10)
        logger.info("Exporting religions to database", job_id=job_id)

        with db.get_session() as session:
            # Export religion definitions
            for religion_id, religion_data in religions_dict.items():
                religion_record = Religion(
                    map_id=map_id,
                    religion_index=religion_id,
                    name=religion_data.name,
                    color=religion_data.color,
                    type=religion_data.type,
                    form=religion_data.form,
                    deity=religion_data.deity
                    if hasattr(religion_data, "deity")
                    else None,
                    expansion=religion_data.expansion
                    if hasattr(religion_data, "expansion")
                    else "global",
                    expansionism=float(religion_data.expansionism),
                    code=religion_data.code
                    if hasattr(religion_data, "code")
                    else f"REL{religion_id}",
                    center_cell_index=religion_data.center,
                    origins=religion_data.origins
                    if hasattr(religion_data, "origins")
                    else [],
                    rural_population=0.0,  # Will be calculated from cells
                    urban_population=0.0,  # Will be calculated from cells
                    cells_count=0,  # Will be calculated from cells
                    area_km2=0.0,  # Will be calculated from cells
                )
                session.add(religion_record)
                session.flush()  # Get the religion ID

                # Export cell assignments for this religion
                religion_cells = np.where(cell_religions == religion_id)[0]
                total_rural_pop = 0.0

                for cell_idx in religion_cells:
                    if cell_idx < len(packed_graph.points):
                        cell_religion = CellReligion(
                            map_id=map_id,
                            cell_index=int(cell_idx),
                            religion_id=religion_record.id,
                            conversion_cost=0.0,  # Default cost
                            dominance_score=1.0,  # Default dominance
                        )
                        session.add(cell_religion)

                        # Add to rural population (using cell population if available)
                        if hasattr(packed_graph, "cell_population") and cell_idx < len(
                            packed_graph.cell_population
                        ):
                            total_rural_pop += float(
                                packed_graph.cell_population[cell_idx]
                            )

                # Update religion statistics
                religion_record.cells_count = len(religion_cells)
                religion_record.rural_population = float(total_rural_pop)

                # Create a simple polygon geometry from religion cells (convex hull)
                if len(religion_cells) >= 3:
                    religion_points = [
                        packed_graph.points[i]
                        for i in religion_cells
                        if i < len(packed_graph.points)
                    ]
                    if len(religion_points) >= 3:
                        try:
                            from scipy.spatial import ConvexHull

                            hull = ConvexHull(religion_points)
                            hull_points = [religion_points[i] for i in hull.vertices]
                            polygon = Polygon(hull_points)
                            religion_record.geometry = from_shape(polygon, srid=4326)
                            religion_record.area_km2 = float(polygon.area)
                        except:
                            # Fallback: create a simple polygon around center
                            center_point = packed_graph.points[religion_data.center]
                            radius = 30.0  # Default radius
                            from shapely.geometry import Point

                            circle = Point(center_point[0], center_point[1]).buffer(
                                radius
                            )
                            religion_record.geometry = from_shape(circle, srid=4326)
                            religion_record.area_km2 = float(circle.area)

            session.commit()
            logger.info(
                f"Exported {len(religions_dict)} religions and {np.sum(cell_religions >= 0)} religion-cell assignments to database"
            )

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

        # Calculate reasonable number of states based on map size
        map_area = request.width * request.height
        reasonable_states = max(3, min(15, int(map_area / 200000)))
        logger.info(
            "Calculated states for map", map_area=map_area, states=reasonable_states
        )

        # Create settlement options with dynamic state count
        from ..core.settlements import SettlementOptions

        settlement_options = SettlementOptions(
            states_number=reasonable_states,
            manors_number=1000,  # Keep default for towns
            growth_rate=1.0,
            states_growth_rate=1.0,
        )

        settlements = Settlements(
            packed_graph,
            features,
            cultures_wrapper,
            biome_wrapper,
            name_generator,
            options=settlement_options,
            cell_religions=cell_religions,
        )
        settlements.generate()

        # Assign temples based on religion system
        religion_generator.assign_temples_to_settlements(settlements.settlements)

        # Update progress
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.progress_percent = 90
            session.commit()

        # Export settlements to database (right after Stage 11)
        logger.info("Exporting settlements to database", job_id=job_id)

        from shapely.geometry import Point
        from geoalchemy2.shape import from_shape

        with db.get_session() as session:
            for settlement_id, settlement_data in settlements.settlements.items():
                # Get settlement position
                cell_idx = settlement_data.cell_id
                if cell_idx < len(packed_graph.points):
                    x, y = packed_graph.points[cell_idx]
                    point_geom = Point(x, y)

                    # Determine settlement type and properties
                    settlement_type = "village"  # Default
                    is_capital = False
                    is_port = False
                    enhanced_type = "Generic Settlement"

                    # Check if this is a capital
                    if settlement_data.is_capital:
                        is_capital = True
                        settlement_type = "capital"
                        enhanced_type = "Capital City"
                    elif settlement_data.population > 10000:
                        settlement_type = "city"
                        enhanced_type = "Large City"
                    elif settlement_data.population > 5000:
                        settlement_type = "town"
                        enhanced_type = "Town"

                    # Check if this is a port
                    if settlement_data.port_id > 0:
                        is_port = True
                        enhanced_type = f"Port {settlement_type.title()}"

                    # Get culture and religion IDs from database records (if available)
                    culture_db_id = None
                    religion_db_id = None

                    if settlement_data.culture_id >= 0:
                        # Find the culture record in database
                        culture_record = (
                            session.query(Culture)
                            .filter(
                                Culture.map_id == map_id,
                                Culture.culture_index
                                == int(settlement_data.culture_id),
                            )
                            .first()
                        )
                        if culture_record:
                            culture_db_id = culture_record.id

                    if settlement_data.religion_id >= 0:
                        # Find the religion record in database
                        religion_record = (
                            session.query(Religion)
                            .filter(
                                Religion.map_id == map_id,
                                Religion.religion_index
                                == int(settlement_data.religion_id),
                            )
                            .first()
                        )
                        if religion_record:
                            religion_db_id = religion_record.id

                    # Create settlement record
                    settlement_record = Settlement(
                        map_id=map_id,
                        settlement_index=settlement_id,
                        name=settlement_data.name,
                        settlement_type=settlement_type,
                        enhanced_type=enhanced_type,
                        population=int(settlement_data.population),
                        geometry=from_shape(point_geom, srid=4326),
                        cell_index=cell_idx,
                        is_capital=is_capital,
                        is_port=is_port,
                        culture_id=culture_db_id,
                        religion_id=religion_db_id,
                        # Architectural features (directly access attributes)
                        citadel=settlement_data.citadel,
                        plaza=settlement_data.plaza,
                        walls=settlement_data.walls,
                        shanty=settlement_data.shanty,
                        temple=settlement_data.temple,
                    )
                    session.add(settlement_record)

            session.commit()
            logger.info(
                f"Exported {len(settlements.settlements)} settlements to database"
            )

        # Stage 12: Finalize map generation (100% progress)

        # Update generation time and complete the job
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            map_obj = session.query(Map).filter(Map.id == map_id).first()

            # Calculate actual generation time (placeholder for now)
            generation_time = 1.0  # TODO: Calculate actual time
            map_obj.generation_time_seconds = generation_time

            # Complete the job
            job.status = "completed"
            job.progress_percent = 100
            job.completed_at = datetime.utcnow()
            session.commit()

        logger.info("Map generation completed", job_id=job_id, map_id=str(map_id))

    except Exception as e:
        logger.error("Map generation failed", job_id=job_id, error=str(e))

        # Update job with error
        with db.get_session() as session:
            job = (
                session.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            )
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.commit()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
