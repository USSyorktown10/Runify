from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete, get_optional_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.schemas.common import SuccessResponse
from app.schemas.route import (
    CreateRouteRequest,
    DetailedRoute,
    PaginatedRoutesResponse,
    UpdateRouteRequest,
)
from app.services.route_service import route_service

router = APIRouter(tags=["routes"])


@router.get("/athletes/{athlete_id}/routes", response_model=PaginatedRoutesResponse)
def list_routes(
    athlete_id: str,
    query: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    stmt = route_service.list_routes(db, athlete_id, query)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    visible = [r for r in items if not r.is_private or (viewer and viewer.id == athlete_id)]
    return PaginatedRoutesResponse(
        pagination=pagination,
        items=[route_service.to_summary(r) for r in visible],
    )


@router.post("/routes")
def create_route(
    body: CreateRouteRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    route = route_service.create(db, athlete.id, body.model_dump())
    return {"route": route, "success": True}


@router.get("/routes/{route_id}", response_model=DetailedRoute)
def get_route(
    route_id: str,
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    return route_service.get(db, route_id, viewer.id if viewer else None)


@router.patch("/routes/{route_id}", response_model=SuccessResponse)
def update_route(
    route_id: str,
    body: UpdateRouteRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = route_service.update(db, route_id, athlete.id, body.model_dump(exclude_unset=True))
    return SuccessResponse(success=success, error_message=error)


@router.delete("/routes/{route_id}", response_model=SuccessResponse)
def delete_route(
    route_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = route_service.delete(db, route_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.get("/routes/{route_id}/export")
def export_route(route_id: str, format: str = Query("gpx"), db: Session = Depends(get_db)):
    gpx = route_service.export_gpx(db, route_id)
    return PlainTextResponse(gpx, media_type="application/gpx+xml")
