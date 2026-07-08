from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.schemas.common import SuccessResponse
from app.schemas.notification import (
    MarkReadRequest,
    PaginatedNotificationsResponse,
    UnreadCountResponse,
)
from app.services.notification_service import notification_service

router = APIRouter(tags=["notifications"])


@router.get("/athlete/notifications", response_model=PaginatedNotificationsResponse)
def list_notifications(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = notification_service.list_notifications(db, athlete.id, page, per_page)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedNotificationsResponse(
        pagination=pagination,
        items=notification_service.to_schemas(db, items),
    )


@router.get("/athlete/notifications/number", response_model=UnreadCountResponse)
def unread_count(athlete: Athlete = Depends(get_current_athlete), db: Session = Depends(get_db)):
    return UnreadCountResponse(unread_count=notification_service.unread_count(db, athlete.id))


@router.post("/athlete/notifications/read", response_model=SuccessResponse)
def mark_read(
    body: MarkReadRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = notification_service.mark_read(db, athlete.id, body.notification_ids)
    return SuccessResponse(success=success, error_message=error)


@router.post("/athlete/notifications/read-all", response_model=SuccessResponse)
def mark_all_read(athlete: Athlete = Depends(get_current_athlete), db: Session = Depends(get_db)):
    success, error = notification_service.mark_all_read(db, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/athlete/notifications/{notification_id}", response_model=SuccessResponse)
def delete_notification(
    notification_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = notification_service.delete(db, athlete.id, notification_id)
    return SuccessResponse(success=success, error_message=error)
