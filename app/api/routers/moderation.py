from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.schemas.common import SuccessResponse
from app.schemas.social import PaginatedAthletesResponse
from app.services.athlete_service import to_summary
from app.services.integration_service import integration_service
from app.services.social_service import social_service

router = APIRouter(tags=["moderation"])


@router.post("/activities/{activity_id}/report", response_model=SuccessResponse)
def report_activity(
    activity_id: str,
    reason: str = Query(...),
    details: str = Query(""),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = integration_service.create_report(db, athlete.id, "activity", activity_id, reason, details)
    return SuccessResponse(success=success, error_message=error)


@router.get("/athlete/blocks", response_model=PaginatedAthletesResponse)
def list_blocks(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_blocks(db, athlete.id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/athletes/{athlete_id}/block", response_model=SuccessResponse)
def block_athlete(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.block(db, athlete.id, athlete_id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/athletes/{athlete_id}/block", response_model=SuccessResponse)
def unblock_athlete(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.unblock(db, athlete.id, athlete_id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/athletes/{athlete_id}/report", response_model=SuccessResponse)
def report_athlete(
    athlete_id: str,
    reason: str = Query(...),
    details: str = Query(""),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = integration_service.create_report(db, athlete.id, "athlete", athlete_id, reason, details)
    return SuccessResponse(success=success, error_message=error)


@router.post("/clubs/{club_id}/report", response_model=SuccessResponse)
def report_club(
    club_id: str,
    reason: str = Query(...),
    details: str = Query(""),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = integration_service.create_report(db, athlete.id, "club", club_id, reason, details)
    return SuccessResponse(success=success, error_message=error)
