from pydantic import BaseModel

from app.schemas.common import PaginatedResponseMetadata


class RouteWaypoint(BaseModel):
    lat: float
    lng: float
    elevation: float
    name: str | None = None


class SummaryRoute(BaseModel):
    id: str
    name: str
    distance: float
    elevation_gain: float
    polyline_summary: str
    is_private: bool
    created_at: str


class DetailedRoute(BaseModel):
    id: str
    athlete_id: str
    name: str
    description: str
    distance: float
    elevation_gain: float
    polyline: str
    waypoints: list[RouteWaypoint]
    is_private: bool
    created_at: str
    estimated_duration: int | None = None


class CreateRouteRequest(BaseModel):
    name: str
    description: str | None = None
    activity_type: str = "run"
    polyline: str | None = None
    activity_id: str | None = None
    waypoints: list[RouteWaypoint] | None = None
    is_private: bool = False


class UpdateRouteRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    is_private: bool | None = None


class PaginatedRoutesResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SummaryRoute]
