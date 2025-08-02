"""
PostGIS query utilities for FMG data.

This module provides high-performance spatial queries for the RPG game engine,
leveraging PostGIS's spatial indexing capabilities.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import structlog
from shapely.geometry import Point, Polygon
from sqlalchemy import and_, or_, text, func
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
)

logger = structlog.get_logger()


class PostGISQueries:
    """
    High-performance spatial queries for RPG game engine.

    Provides optimized queries for common game operations like:
    - Finding settlements within a region
    - Getting territorial information at a point
    - Querying river networks
    - Climate and biome data lookup
    """

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def get_map_by_id(self, map_id: uuid.UUID) -> Optional[Map]:
        """Get map record by ID."""
        return self.session.query(Map).filter(Map.id == map_id).first()

    def get_map_by_name(self, map_name: str) -> Optional[Map]:
        """Get map record by name."""
        return self.session.query(Map).filter(Map.name == map_name).first()

    def list_maps(self, limit: int = 50) -> List[Map]:
        """List all maps with metadata."""
        return (
            self.session.query(Map).order_by(Map.created_at.desc()).limit(limit).all()
        )

    def get_territories_at_point(
        self, map_id: uuid.UUID, longitude: float, latitude: float
    ) -> Dict[str, Any]:
        """
        Get all territorial information at a specific point.

        Returns culture, religion, state, and biome data for the given coordinates.
        Optimized for < 50ms response time using spatial indexes.

        Args:
            map_id: Map UUID
            longitude: X coordinate (WGS84)
            latitude: Y coordinate (WGS84)

        Returns:
            Dictionary with territorial information
        """
        point = func.ST_SetSRID(func.ST_Point(longitude, latitude), 4326)

        result = {
            "coordinates": {"longitude": longitude, "latitude": latitude},
            "culture": None,
            "religion": None,
            "state": None,
            "biome": None,
        }

        try:
            # Query culture
            culture = (
                self.session.query(Culture)
                .filter(
                    and_(
                        Culture.map_id == map_id,
                        func.ST_Contains(Culture.geometry, point),
                    )
                )
                .first()
            )

            if culture:
                result["culture"] = {
                    "id": str(culture.id),
                    "name": culture.name,
                    "type": culture.type,
                    "color": culture.color,
                    "expansionism": culture.expansionism,
                    "population": culture.population,
                }

            # Query religion
            religion = (
                self.session.query(Religion)
                .filter(
                    and_(
                        Religion.map_id == map_id,
                        func.ST_Contains(Religion.geometry, point),
                    )
                )
                .first()
            )

            if religion:
                result["religion"] = {
                    "id": str(religion.id),
                    "name": religion.name,
                    "type": religion.type,
                    "form": religion.form,
                    "color": religion.color,
                    "deity": religion.deity,
                }

            # Query state
            state = (
                self.session.query(State)
                .filter(
                    and_(
                        State.map_id == map_id, func.ST_Contains(State.geometry, point)
                    )
                )
                .first()
            )

            if state:
                result["state"] = {
                    "id": str(state.id),
                    "name": state.name,
                    "government_type": state.government_type,
                    "state_type": state.state_type,
                    "color": state.color,
                    "population": state.population,
                }

            # Query biome
            biome = (
                self.session.query(BiomeRegion)
                .filter(
                    and_(
                        BiomeRegion.map_id == map_id,
                        func.ST_Contains(BiomeRegion.geometry, point),
                    )
                )
                .first()
            )

            if biome:
                result["biome"] = {
                    "id": str(biome.id),
                    "type": biome.biome_type,
                    "classification": biome.biome_classification,
                    "habitability": biome.habitability_score,
                    "temperature": biome.avg_temperature_c,
                    "precipitation": biome.avg_precipitation_mm,
                }

            return result

        except Exception as e:
            logger.error("Failed to query territories at point", error=str(e))
            return result

    def find_settlements_in_region(
        self,
        map_id: uuid.UUID,
        bbox: Tuple[float, float, float, float],
        settlement_types: Optional[List[str]] = None,
        min_population: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Find settlements within a bounding box.

        Args:
            map_id: Map UUID
            bbox: (min_lon, min_lat, max_lon, max_lat)
            settlement_types: Filter by settlement types
            min_population: Minimum population filter
            limit: Maximum results to return

        Returns:
            List of settlement dictionaries
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_geom = func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)

        query = self.session.query(Settlement).filter(
            and_(
                Settlement.map_id == map_id,
                func.ST_Intersects(Settlement.geometry, bbox_geom),
            )
        )

        if settlement_types:
            query = query.filter(Settlement.settlement_type.in_(settlement_types))

        if min_population:
            query = query.filter(Settlement.population >= min_population)

        settlements = query.order_by(Settlement.population.desc()).limit(limit).all()

        result = []
        for settlement in settlements:
            # Extract coordinates from geometry
            coords = self.session.scalar(
                func.ST_X(settlement.geometry).label("x"),
                func.ST_Y(settlement.geometry).label("y"),
            )

            result.append(
                {
                    "id": str(settlement.id),
                    "name": settlement.name,
                    "type": settlement.settlement_type,
                    "population": settlement.population,
                    "coordinates": {
                        "longitude": float(coords[0]) if coords else 0,
                        "latitude": float(coords[1]) if coords else 0,
                    },
                    "is_capital": settlement.is_capital,
                    "is_port": settlement.is_port,
                    "features": {
                        "citadel": settlement.citadel,
                        "plaza": settlement.plaza,
                        "walls": settlement.walls,
                        "temple": settlement.temple,
                    },
                }
            )

        return result

    def get_river_network(
        self,
        map_id: uuid.UUID,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        min_discharge: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get river network data.

        Args:
            map_id: Map UUID
            bbox: Optional bounding box filter
            min_discharge: Minimum discharge filter (m³/s)

        Returns:
            List of river dictionaries with geometries
        """
        query = self.session.query(River).filter(River.map_id == map_id)

        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            bbox_geom = func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
            query = query.filter(func.ST_Intersects(River.geometry, bbox_geom))

        if min_discharge:
            query = query.filter(River.discharge_m3s >= min_discharge)

        rivers = query.order_by(River.discharge_m3s.desc()).all()

        result = []
        for river in rivers:
            # Convert geometry to GeoJSON-like format
            geom_wkt = self.session.scalar(func.ST_AsText(river.geometry))

            result.append(
                {
                    "id": str(river.id),
                    "name": river.name,
                    "geometry_wkt": geom_wkt,
                    "length_km": river.length_km,
                    "discharge_m3s": river.discharge_m3s,
                    "width_m": river.average_width_m,
                    "is_main_stem": river.is_main_stem,
                }
            )

        return result

    def get_climate_data(
        self,
        map_id: uuid.UUID,
        longitude: float,
        latitude: float,
        radius_km: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get climate data for a location with optional radius averaging.

        Args:
            map_id: Map UUID
            longitude: X coordinate
            latitude: Y coordinate
            radius_km: Radius for averaging climate data

        Returns:
            Climate data dictionary
        """
        point = func.ST_SetSRID(func.ST_Point(longitude, latitude), 4326)

        # Query climate data within radius
        climate_query = self.session.query(
            func.avg(ClimateData.temperature_c).label("avg_temp"),
            func.avg(ClimateData.precipitation_mm).label("avg_precip"),
            func.count(ClimateData.id).label("sample_count"),
        ).filter(
            and_(
                ClimateData.map_id == map_id,
                func.ST_DWithin(
                    func.ST_SetSRID(
                        func.ST_Point(ClimateData.latitude, ClimateData.latitude), 4326
                    ),
                    point,
                    radius_km * 1000,  # Convert km to meters
                ),
            )
        )

        result = climate_query.first()

        if result and result.sample_count > 0:
            return {
                "temperature_c": float(result.avg_temp),
                "precipitation_mm": float(result.avg_precip),
                "sample_count": int(result.sample_count),
                "radius_km": radius_km,
            }
        else:
            return {
                "temperature_c": None,
                "precipitation_mm": None,
                "sample_count": 0,
                "radius_km": radius_km,
            }

    def export_to_geopandas(
        self,
        map_id: uuid.UUID,
        table_name: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Export spatial data to GeoPandas DataFrame.

        Args:
            map_id: Map UUID
            table_name: Name of table to export ('cultures', 'states', etc.)
            bbox: Optional bounding box filter

        Returns:
            GeoDataFrame with spatial data
        """
        table_map = {
            "cultures": Culture,
            "religions": Religion,
            "states": State,
            "settlements": Settlement,
            "rivers": River,
            "biomes": BiomeRegion,
        }

        if table_name not in table_map:
            raise ValueError(f"Unknown table: {table_name}")

        model = table_map[table_name]
        query = self.session.query(model).filter(model.map_id == map_id)

        if bbox and hasattr(model, "geometry"):
            min_lon, min_lat, max_lon, max_lat = bbox
            bbox_geom = func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
            query = query.filter(func.ST_Intersects(model.geometry, bbox_geom))

        # Use GeoAlchemy2 to load geometries directly into GeoDataFrame
        df = gpd.read_postgis(
            sql=query.statement, con=self.session.bind, geom_col="geometry"
        )

        return df

    def get_map_statistics(self, map_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a map.

        Args:
            map_id: Map UUID

        Returns:
            Dictionary with map statistics
        """
        stats = {}

        try:
            # Basic counts
            stats["cultures_count"] = (
                self.session.query(Culture).filter(Culture.map_id == map_id).count()
            )

            stats["religions_count"] = (
                self.session.query(Religion).filter(Religion.map_id == map_id).count()
            )

            stats["states_count"] = (
                self.session.query(State).filter(State.map_id == map_id).count()
            )

            stats["settlements_count"] = (
                self.session.query(Settlement)
                .filter(Settlement.map_id == map_id)
                .count()
            )

            stats["rivers_count"] = (
                self.session.query(River).filter(River.map_id == map_id).count()
            )

            stats["biomes_count"] = (
                self.session.query(BiomeRegion)
                .filter(BiomeRegion.map_id == map_id)
                .count()
            )

            # Population statistics
            total_pop = (
                self.session.query(func.sum(Settlement.population))
                .filter(Settlement.map_id == map_id)
                .scalar()
            )

            stats["total_population"] = int(total_pop) if total_pop else 0

            # Largest settlement
            largest_settlement = (
                self.session.query(Settlement)
                .filter(Settlement.map_id == map_id)
                .order_by(Settlement.population.desc())
                .first()
            )

            if largest_settlement:
                stats["largest_settlement"] = {
                    "name": largest_settlement.name,
                    "population": largest_settlement.population,
                    "type": largest_settlement.settlement_type,
                }

            # Territory areas
            total_land_area = (
                self.session.query(func.sum(func.ST_Area(Culture.geometry)))
                .filter(Culture.map_id == map_id)
                .scalar()
            )

            stats["total_land_area_km2"] = (
                float(total_land_area) if total_land_area else 0
            )

            return stats

        except Exception as e:
            logger.error("Failed to get map statistics", error=str(e))
            return stats


def create_query_helper(session: Session) -> PostGISQueries:
    """Create a PostGIS query helper instance."""
    return PostGISQueries(session)
