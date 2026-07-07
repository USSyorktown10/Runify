import polyline as polyline_lib
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ForbiddenError, NotFoundError
from app.models.activity import Activity
from app.models.segment import Route, RouteWaypoint
from app.schemas.route import DetailedRoute, SummaryRoute
from app.schemas.route import RouteWaypoint as RouteWaypointSchema


class RouteService:
    def to_summary(self, route: Route) -> SummaryRoute:
        return SummaryRoute(
            id=route.id,
            name=route.name,
            distance=route.distance,
            elevation_gain=route.elevation_gain,
            polyline_summary=route.polyline_summary,
            is_private=route.is_private,
            created_at=route.created_at.isoformat(),
        )

    def to_detailed(self, db: Session, route: Route) -> DetailedRoute:
        waypoints = db.scalars(
            select(RouteWaypoint).where(RouteWaypoint.route_id == route.id).order_by(RouteWaypoint.sequence)
        ).all()
        return DetailedRoute(
            id=route.id,
            athlete_id=route.athlete_id,
            name=route.name,
            description=route.description,
            distance=route.distance,
            elevation_gain=route.elevation_gain,
            polyline=route.polyline,
            waypoints=[
                RouteWaypointSchema(lat=w.lat, lng=w.lng, elevation=w.elevation, name=w.name) for w in waypoints
            ],
            is_private=route.is_private,
            created_at=route.created_at.isoformat(),
            estimated_duration=route.estimated_duration,
        )

    def list_routes(self, db: Session, athlete_id: str, query: str | None):
        stmt = select(Route).where(Route.athlete_id == athlete_id)
        if query:
            stmt = stmt.where(Route.name.ilike(f"%{query}%"))
        return stmt.order_by(Route.created_at.desc())

    def create(self, db: Session, athlete_id: str, data: dict) -> DetailedRoute:
        poly = data.get("polyline", "")
        distance = 0.0
        if data.get("activity_id"):
            activity = db.get(Activity, data["activity_id"])
            if activity:
                poly = activity.polyline
                distance = activity.distance
        if poly:
            coords = polyline_lib.decode(poly)
            if len(coords) >= 2:
                from app.services.metrics_engine.formulas import haversine
                for i in range(1, len(coords)):
                    distance += haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        route = Route(
            athlete_id=athlete_id,
            name=data["name"],
            description=data.get("description", ""),
            activity_type=data.get("activity_type", "run"),
            polyline=poly,
            polyline_summary=poly,
            distance=distance,
            is_private=data.get("is_private", False),
        )
        db.add(route)
        db.flush()
        for i, wp in enumerate(data.get("waypoints") or []):
            db.add(
                RouteWaypoint(
                    route_id=route.id,
                    lat=wp["lat"] if isinstance(wp, dict) else wp.lat,
                    lng=wp["lng"] if isinstance(wp, dict) else wp.lng,
                    elevation=wp.get("elevation", 0) if isinstance(wp, dict) else wp.elevation,
                    name=wp.get("name") if isinstance(wp, dict) else wp.name,
                    sequence=i,
                )
            )
        db.commit()
        db.refresh(route)
        return self.to_detailed(db, route)

    def get(self, db: Session, route_id: str, viewer_id: str | None) -> DetailedRoute:
        route = db.get(Route, route_id)
        if not route:
            raise NotFoundError("Route not found")
        if route.is_private and route.athlete_id != viewer_id:
            raise ForbiddenError("Route is private")
        return self.to_detailed(db, route)

    def update(self, db: Session, route_id: str, athlete_id: str, data: dict) -> tuple[bool, str | None]:
        route = db.get(Route, route_id)
        if not route or route.athlete_id != athlete_id:
            return False, "Route not found"
        for field in ("name", "description", "is_private"):
            if field in data and data[field] is not None:
                setattr(route, field, data[field])
        db.commit()
        return True, None

    def delete(self, db: Session, route_id: str, athlete_id: str) -> tuple[bool, str | None]:
        route = db.get(Route, route_id)
        if not route or route.athlete_id != athlete_id:
            return False, "Route not found"
        db.delete(route)
        db.commit()
        return True, None

    def export_gpx(self, db: Session, route_id: str) -> str:
        route = db.get(Route, route_id)
        if not route:
            raise NotFoundError()
        coords = polyline_lib.decode(route.polyline) if route.polyline else []
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="Runify">',
            "  <trk><trkseg>",
        ]
        for lat, lng in coords:
            lines.append(f'    <trkpt lat="{lat}" lon="{lng}"></trkpt>')
        lines.extend(["  </trkseg></trk>", "</gpx>"])
        return "\n".join(lines)


route_service = RouteService()
