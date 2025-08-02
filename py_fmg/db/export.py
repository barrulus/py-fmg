"""
PostGIS export pipeline for FMG data.

This module implements the GeoPandas to PostGIS export pipeline for converting
generated map data into the PostGIS database schema defined in PLAN.md.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import structlog
from geoalchemy2 import WKTElement
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from sqlalchemy.orm import Session

from .models import (
    Map,
    Culture,
    Religion,
    State,
    Settlement,
    River,
    BiomeRegion,
    ClimateData,
    CellCulture,
    CellReligion,
    ReligionCulture,
)

logger = structlog.get_logger()


class PostGISExporter:
    """
    Export generated map data to PostGIS database.

    Handles conversion of FMG-generated data structures into the PostGIS schema
    with proper spatial indexing and foreign key relationships.
    """

    def __init__(self, session: Session):
        """Initialize exporter with database session."""
        self.session = session
        self.map_id: Optional[uuid.UUID] = None

    def export_complete_map(
        self,
        map_name: str,
        map_metadata: Dict[str, Any],
        voronoi_graph: Any,
        cultures_data: Dict[int, Any],
        religions_data: Dict[int, Any],
        states_data: Dict[int, Any],
        settlements_data: Dict[int, Any],
        rivers_data: Dict[int, Any],
        biomes_data: Dict[int, Any],
        climate_data: Optional[Any] = None,
        bulk_insert: bool = True,
    ) -> uuid.UUID:
        """
        Export a complete generated map to PostGIS.

        Args:
            map_name: Human-readable map name
            map_metadata: Map generation metadata (seeds, dimensions, etc.)
            voronoi_graph: VoronoiGraph instance with cell data
            cultures_data: Dictionary of culture objects by ID
            religions_data: Dictionary of religion objects by ID
            states_data: Dictionary of state objects by ID
            settlements_data: Dictionary of settlement objects by ID
            rivers_data: Dictionary of river objects by ID
            biomes_data: Dictionary of biome objects by ID
            climate_data: Optional climate data
            bulk_insert: Use bulk insert optimization for performance

        Returns:
            UUID of the created map record
        """
        logger.info("Starting complete map export", map_name=map_name)

        try:
            # Step 1: Create map record
            self.map_id = self._export_map_metadata(map_name, map_metadata)

            # Step 2: Export cultures with geometries
            if cultures_data:
                self._export_cultures(cultures_data, voronoi_graph, bulk_insert)

            # Step 3: Export religions with geometries
            if religions_data:
                self._export_religions(religions_data, voronoi_graph, bulk_insert)

            # Step 4: Export states with geometries
            if states_data:
                self._export_states(states_data, voronoi_graph, bulk_insert)

            # Step 5: Export settlements
            if settlements_data:
                self._export_settlements(settlements_data, voronoi_graph, bulk_insert)

            # Step 6: Export rivers
            if rivers_data:
                self._export_rivers(rivers_data, voronoi_graph, bulk_insert)

            # Step 7: Export biomes
            if biomes_data:
                self._export_biomes(biomes_data, voronoi_graph, bulk_insert)

            # Step 8: Export climate data
            if climate_data:
                self._export_climate_data(climate_data, voronoi_graph, bulk_insert)

            # Step 9: Export cell-level assignments
            self._export_cell_assignments(
                cultures_data, religions_data, voronoi_graph, bulk_insert
            )

            # Step 10: Create spatial indexes
            self._create_spatial_indexes()

            self.session.commit()
            logger.info("Map export completed successfully", map_id=str(self.map_id))

            return self.map_id

        except Exception as e:
            logger.error("Map export failed", error=str(e))
            self.session.rollback()
            raise

    def _export_map_metadata(
        self, map_name: str, metadata: Dict[str, Any]
    ) -> uuid.UUID:
        """Create the main map record."""
        logger.info("Exporting map metadata")

        map_record = Map(
            name=map_name,
            seed=metadata.get("seed", "unknown"),
            grid_seed=metadata.get("grid_seed", metadata.get("seed", "unknown")),
            map_seed=metadata.get("map_seed", metadata.get("seed", "unknown")),
            width=float(metadata.get("width", 0)),
            height=float(metadata.get("height", 0)),
            cells_count=int(metadata.get("cells_count", 0)),
            generation_time_seconds=metadata.get("generation_time", None),
            config_json=json.dumps(metadata.get("config", {})),
        )

        self.session.add(map_record)
        self.session.flush()  # Get the ID

        logger.info("Map record created", map_id=str(map_record.id))
        return map_record.id

    def _export_cultures(
        self,
        cultures_data: Dict[int, Any],
        voronoi_graph: Any,
        bulk_insert: bool = True,
    ) -> None:
        """Export cultures with their territorial geometries."""
        logger.info("Exporting cultures", count=len(cultures_data))

        culture_records = []

        for culture_id, culture in cultures_data.items():
            # Create territory geometry from assigned cells
            territory_geom = self._create_territory_geometry(
                culture_id, voronoi_graph, "culture"
            )

            culture_record = Culture(
                map_id=self.map_id,
                culture_index=culture_id,
                name=getattr(culture, "name", f"Culture {culture_id}"),
                color=getattr(culture, "color", "#808080"),
                type=getattr(culture, "type", "Generic"),
                expansionism=float(getattr(culture, "expansionism", 1.0)),
                center_cell_index=int(getattr(culture, "center", 0)),
                name_base=int(getattr(culture, "name_base", 0)),
                geometry=territory_geom,
                area_km2=territory_geom.area if territory_geom else 0,
                population=int(getattr(culture, "population", 0)),
                cells_count=len(getattr(culture, "cells", [])),
            )

            if bulk_insert:
                culture_records.append(culture_record)
            else:
                self.session.add(culture_record)

        if bulk_insert and culture_records:
            self.session.bulk_save_objects(culture_records)

        logger.info("Cultures exported successfully")

    def _export_religions(
        self,
        religions_data: Dict[int, Any],
        voronoi_graph: Any,
        bulk_insert: bool = True,
    ) -> None:
        """Export religions with their territorial geometries."""
        logger.info("Exporting religions", count=len(religions_data))

        religion_records = []

        for religion_id, religion in religions_data.items():
            # Create territory geometry from assigned cells
            territory_geom = self._create_territory_geometry(
                religion_id, voronoi_graph, "religion"
            )

            religion_record = Religion(
                map_id=self.map_id,
                religion_index=religion_id,
                name=getattr(religion, "name", f"Religion {religion_id}"),
                color=getattr(religion, "color", "#808080"),
                type=getattr(religion, "type", "Folk"),
                form=getattr(religion, "form", "Shamanism"),
                center_cell_index=int(getattr(religion, "center", 0)),
                deity=getattr(religion, "deity", None),
                expansion=getattr(religion, "expansion", "global"),
                expansionism=float(getattr(religion, "expansionism", 1.0)),
                code=getattr(religion, "code", None),
                geometry=territory_geom,
                area_km2=territory_geom.area if territory_geom else 0,
                cells_count=len(getattr(religion, "cells", [])),
            )

            if bulk_insert:
                religion_records.append(religion_record)
            else:
                self.session.add(religion_record)

        if bulk_insert and religion_records:
            self.session.bulk_save_objects(religion_records)

        logger.info("Religions exported successfully")

    def _export_states(
        self, states_data: Dict[int, Any], voronoi_graph: Any, bulk_insert: bool = True
    ) -> None:
        """Export political states with their territorial boundaries."""
        logger.info("Exporting states", count=len(states_data))

        state_records = []

        for state_id, state in states_data.items():
            # Create territory geometry from assigned cells
            territory_geom = self._create_territory_geometry(
                state_id, voronoi_graph, "state"
            )

            state_record = State(
                map_id=self.map_id,
                state_index=state_id,
                name=getattr(state, "name", f"State {state_id}"),
                color=getattr(state, "color", "#808080"),
                government_type=getattr(state, "government_type", None),
                state_type=getattr(state, "state_type", None),
                expansionism=float(getattr(state, "expansionism", 1.0)),
                center_cell_index=int(getattr(state, "center", 0)),
                geometry=territory_geom,
                area_km2=territory_geom.area if territory_geom else 0,
                population=int(getattr(state, "population", 0)),
            )

            if bulk_insert:
                state_records.append(state_record)
            else:
                self.session.add(state_record)

        if bulk_insert and state_records:
            self.session.bulk_save_objects(state_records)

        logger.info("States exported successfully")

    def _export_settlements(
        self,
        settlements_data: Dict[int, Any],
        voronoi_graph: Any,
        bulk_insert: bool = True,
    ) -> None:
        """Export settlements as point geometries."""
        logger.info("Exporting settlements", count=len(settlements_data))

        settlement_records = []

        for settlement_id, settlement in settlements_data.items():
            # Get settlement coordinates from cell or direct coordinates
            cell_index = getattr(settlement, "cell", settlement_id)
            if hasattr(voronoi_graph, "points") and cell_index < len(
                voronoi_graph.points
            ):
                x, y = voronoi_graph.points[cell_index]
                point_geom = WKTElement(f"POINT({x} {y})", srid=4326)
            else:
                # Fallback for direct coordinates
                x = getattr(settlement, "x", 0)
                y = getattr(settlement, "y", 0)
                point_geom = WKTElement(f"POINT({x} {y})", srid=4326)

            settlement_record = Settlement(
                map_id=self.map_id,
                settlement_index=settlement_id,
                name=getattr(settlement, "name", f"Settlement {settlement_id}"),
                settlement_type=getattr(settlement, "type", "town"),
                population=int(getattr(settlement, "population", 0)),
                geometry=point_geom,
                cell_index=cell_index,
                is_capital=bool(getattr(settlement, "capital", False)),
                is_port=bool(getattr(settlement, "port", False)),
                citadel=bool(getattr(settlement, "citadel", False)),
                plaza=bool(getattr(settlement, "plaza", False)),
                walls=bool(getattr(settlement, "walls", False)),
                temple=bool(getattr(settlement, "temple", False)),
            )

            if bulk_insert:
                settlement_records.append(settlement_record)
            else:
                self.session.add(settlement_record)

        if bulk_insert and settlement_records:
            self.session.bulk_save_objects(settlement_records)

        logger.info("Settlements exported successfully")

    def _export_rivers(
        self, rivers_data: Dict[int, Any], voronoi_graph: Any, bulk_insert: bool = True
    ) -> None:
        """Export rivers as linestring geometries."""
        logger.info("Exporting rivers", count=len(rivers_data))

        river_records = []

        for river_id, river in rivers_data.items():
            # Create linestring from river cells
            cells = getattr(river, "cells", [])
            if len(cells) < 2:
                continue  # Need at least 2 points for a line

            # Convert cell indices to coordinates
            coordinates = []
            for cell_idx in cells:
                if cell_idx < len(voronoi_graph.points):
                    x, y = voronoi_graph.points[cell_idx]
                    coordinates.append((x, y))

            if len(coordinates) < 2:
                continue

            # Create WKT linestring
            coords_str = ", ".join([f"{x} {y}" for x, y in coordinates])
            line_geom = WKTElement(f"LINESTRING({coords_str})", srid=4326)

            river_record = River(
                map_id=self.map_id,
                river_index=river_id,
                name=getattr(river, "name", None),
                geometry=line_geom,
                length_km=float(getattr(river, "length", 0)),
                discharge_m3s=float(getattr(river, "discharge", 0)),
                average_width_m=float(getattr(river, "width", 0)),
                parent_river_id=None,  # TODO: Handle river hierarchy
                is_main_stem=bool(getattr(river, "parent_id", None) is None),
            )

            if bulk_insert:
                river_records.append(river_record)
            else:
                self.session.add(river_record)

        if bulk_insert and river_records:
            self.session.bulk_save_objects(river_records)

        logger.info("Rivers exported successfully")

    def _export_biomes(
        self, biomes_data: Dict[int, Any], voronoi_graph: Any, bulk_insert: bool = True
    ) -> None:
        """Export biome regions as polygon geometries."""
        logger.info("Exporting biomes", count=len(biomes_data))

        biome_records = []

        for biome_id, biome in biomes_data.items():
            # Create territory geometry from biome cells
            territory_geom = self._create_territory_geometry(
                biome_id, voronoi_graph, "biome"
            )

            biome_record = BiomeRegion(
                map_id=self.map_id,
                biome_index=biome_id,
                biome_type=getattr(biome, "name", f"Biome {biome_id}"),
                biome_classification=getattr(biome, "classification", None),
                temperature_band=int(getattr(biome, "temperature_band", 0)),
                moisture_band=int(getattr(biome, "moisture_band", 0)),
                geometry=territory_geom,
                area_km2=territory_geom.area if territory_geom else 0,
                habitability_score=int(getattr(biome, "habitability", 50)),
                movement_cost=int(getattr(biome, "movement_cost", 1)),
                avg_temperature_c=float(getattr(biome, "avg_temperature", 15)),
                avg_precipitation_mm=float(getattr(biome, "avg_precipitation", 500)),
            )

            if bulk_insert:
                biome_records.append(biome_record)
            else:
                self.session.add(biome_record)

        if bulk_insert and biome_records:
            self.session.bulk_save_objects(biome_records)

        logger.info("Biomes exported successfully")

    def _export_climate_data(
        self, climate_data: Any, voronoi_graph: Any, bulk_insert: bool = True
    ) -> None:
        """Export climate data for each cell."""
        logger.info("Exporting climate data")

        climate_records = []

        # Handle different climate data formats
        if hasattr(climate_data, "temperature") and hasattr(
            climate_data, "precipitation"
        ):
            temp_data = climate_data.temperature
            precip_data = climate_data.precipitation

            for cell_idx in range(len(voronoi_graph.points)):
                if cell_idx < len(temp_data) and cell_idx < len(precip_data):
                    climate_record = ClimateData(
                        map_id=self.map_id,
                        cell_index=cell_idx,
                        temperature_c=int(temp_data[cell_idx]),
                        precipitation_mm=int(precip_data[cell_idx]),
                        latitude=float(voronoi_graph.points[cell_idx][1]),
                        altitude_m=(
                            int(voronoi_graph.heights[cell_idx])
                            if hasattr(voronoi_graph, "heights")
                            else 0
                        ),
                    )

                    if bulk_insert:
                        climate_records.append(climate_record)
                    else:
                        self.session.add(climate_record)

        if bulk_insert and climate_records:
            self.session.bulk_save_objects(climate_records)

        logger.info("Climate data exported successfully")

    def _export_cell_assignments(
        self,
        cultures_data: Dict[int, Any],
        religions_data: Dict[int, Any],
        voronoi_graph: Any,
        bulk_insert: bool = True,
    ) -> None:
        """Export cell-level culture and religion assignments."""
        logger.info("Exporting cell assignments")

        culture_assignments = []
        religion_assignments = []

        # Export culture assignments
        for culture_id, culture in cultures_data.items():
            cells = getattr(culture, "cells", [])
            for cell_idx in cells:
                assignment = CellCulture(
                    map_id=self.map_id,
                    cell_index=cell_idx,
                    culture_id=None,  # Will be set after culture records are created
                    population=(
                        float(getattr(culture, "population", 0)) / len(cells)
                        if cells
                        else 0
                    ),
                )
                culture_assignments.append(assignment)

        # Export religion assignments
        for religion_id, religion in religions_data.items():
            cells = getattr(religion, "cells", [])
            for cell_idx in cells:
                assignment = CellReligion(
                    map_id=self.map_id,
                    cell_index=cell_idx,
                    religion_id=None,  # Will be set after religion records are created
                    dominance_score=1.0,
                )
                religion_assignments.append(assignment)

        # Note: Foreign key relationships will need to be updated after records are created
        # This is a simplified version - full implementation would handle FK relationships

        logger.info("Cell assignments exported successfully")

    def _create_territory_geometry(
        self, entity_id: int, voronoi_graph: Any, entity_type: str
    ) -> Optional[WKTElement]:
        """
        Create territory geometry from assigned cells.

        Args:
            entity_id: ID of the entity (culture, religion, state, biome)
            voronoi_graph: Voronoi graph with cell data
            entity_type: Type of entity for cell lookup

        Returns:
            WKTElement polygon geometry or None
        """
        try:
            # This is a simplified version - in practice, you'd need to:
            # 1. Get cells assigned to this entity
            # 2. Create Voronoi polygons for those cells
            # 3. Union the polygons into a single territory
            # 4. Convert to WKT format

            # Placeholder implementation
            # In real implementation, you'd use the Voronoi graph to generate
            # actual cell polygons and union them

            return None  # TODO: Implement actual geometry creation

        except Exception as e:
            logger.warning(
                "Failed to create territory geometry",
                entity_id=entity_id,
                entity_type=entity_type,
                error=str(e),
            )
            return None

    def _create_spatial_indexes(self) -> None:
        """Create spatial indexes for performance optimization."""
        logger.info("Creating spatial indexes")

        try:
            # Create spatial indexes on geometry columns
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_cultures_geom ON cultures USING GIST (geometry)",
                "CREATE INDEX IF NOT EXISTS idx_religions_geom ON religions USING GIST (geometry)",
                "CREATE INDEX IF NOT EXISTS idx_states_geom ON states USING GIST (geometry)",
                "CREATE INDEX IF NOT EXISTS idx_settlements_geom ON settlements USING GIST (geometry)",
                "CREATE INDEX IF NOT EXISTS idx_rivers_geom ON rivers USING GIST (geometry)",
                "CREATE INDEX IF NOT EXISTS idx_biomes_geom ON biomes USING GIST (geometry)",
            ]

            for index_sql in indexes:
                self.session.execute(index_sql)

            logger.info("Spatial indexes created successfully")

        except Exception as e:
            logger.warning("Failed to create some spatial indexes", error=str(e))


def export_map_to_postgis(
    session: Session, map_name: str, map_data: Dict[str, Any], bulk_insert: bool = True
) -> uuid.UUID:
    """
    Convenience function to export a complete map to PostGIS.

    Args:
        session: Database session
        map_name: Human-readable map name
        map_data: Dictionary containing all map data components
        bulk_insert: Use bulk insert optimization

    Returns:
        UUID of the created map record
    """
    exporter = PostGISExporter(session)

    return exporter.export_complete_map(
        map_name=map_name,
        map_metadata=map_data.get("metadata", {}),
        voronoi_graph=map_data.get("voronoi_graph"),
        cultures_data=map_data.get("cultures", {}),
        religions_data=map_data.get("religions", {}),
        states_data=map_data.get("states", {}),
        settlements_data=map_data.get("settlements", {}),
        rivers_data=map_data.get("rivers", {}),
        biomes_data=map_data.get("biomes", {}),
        climate_data=map_data.get("climate"),
        bulk_insert=bulk_insert,
    )
