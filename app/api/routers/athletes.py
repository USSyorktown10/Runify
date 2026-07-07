from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete, get_optional_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.schemas.activity import Stream
from app.schemas.athlete import (
    AthletePreferences,
    AthleteStats,
    DetailedAthlete,
    MeAthlete,
    PaginatedConnectionsResponse,
    PersonalRecord,
    UpdateAthleteProfileRequest,
    UpdateAthleteStatsRequest,
    UpdatePreferencesRequest,
)
from app.schemas.common import SuccessResponse
from app.services.athlete_service import athlete_service
from app.services.social_service import social_service

router = APIRouter(tags=["athletes"])


@router.get("/athlete/me", response_model=MeAthlete)
def get_me(athlete: Athlete = Depends(get_current_athlete), db: Session = Depends(get_db)):
    return athlete_service.get_me(db, athlete)


@router.patch("/athlete/profile", response_model=MeAthlete)
def update_profile(
    body: UpdateAthleteProfileRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    return athlete_service.update_profile(db, athlete, body.model_dump(exclude_unset=True))


@router.get("/athletes/search", response_model=PaginatedConnectionsResponse)
def search_athletes(
    query: str = Query(""),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    from app.schemas.athlete import ConnectionSearchResult
    from app.services.athlete_service import to_summary

    stmt = athlete_service.search(db, query, athlete, page, per_page)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    results = [
        ConnectionSearchResult(
            athlete=to_summary(a),
            relationship_status=social_service.relationship_status(db, athlete.id, a.id),
            connection_degree=1,
            common_clubs_count=0,
            mutual_followers_count=0,
        )
        for a in items
    ]
    return PaginatedConnectionsResponse(pagination=pagination, items=results)


@router.get("/athletes/{athlete_id}", response_model=DetailedAthlete)
def get_athlete(
    athlete_id: str,
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    return athlete_service.get_detailed(db, athlete_id, viewer)


@router.get("/athletes/{athlete_id}/records", response_model=list[PersonalRecord])
def get_records(athlete_id: str, db: Session = Depends(get_db)):
    from sqlalchemy import select

    from app.models.athlete import PersonalRecord as PRModel

    records = db.scalars(select(PRModel).where(PRModel.athlete_id == athlete_id)).all()
    return [
        PersonalRecord(
            distance_name=r.distance_name,
            time_in_seconds=r.time_in_seconds,
            activity_id=r.activity_id or "",
            achieved_date=r.achieved_date.isoformat(),
        )
        for r in records
    ]


@router.get("/athletes/{athlete_id}/stats", response_model=AthleteStats)
def get_stats(athlete_id: str, db: Session = Depends(get_db)):
    from app.models.athlete import AthleteStats as StatsModel
    stats = db.query(StatsModel).filter_by(athlete_id=athlete_id).first()
    return AthleteStats(
        current_ftp=stats.current_ftp if stats else 0,
        threshold_pace=stats.threshold_pace if stats else 3.5,
        ytd_run_totals=stats.ytd_run_totals if stats else 0,
        all_time_run_totals=stats.all_time_run_totals if stats else 0,
    )


@router.patch("/athlete/stats")
def update_stats(
    body: UpdateAthleteStatsRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stats = athlete_service.update_stats(db, athlete.id, body.model_dump(exclude_unset=True))
    return {"success": True, **stats.model_dump()}


@router.get("/athletes/{athlete_id}/streams", response_model=list[Stream])
def get_athlete_streams(
    athlete_id: str,
    streams: str = Query(""),
    resolution: str = Query("high"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    db: Session = Depends(get_db),
):
    from datetime import datetime

    from sqlalchemy import select

    from app.models.athlete import AthleteStream

    stmt = select(AthleteStream).where(AthleteStream.athlete_id == athlete_id)
    if streams:
        keys = streams.split(",")
        stmt = stmt.where(AthleteStream.metric_key.in_(keys))
    if start_date:
        stmt = stmt.where(AthleteStream.recorded_at >= datetime.fromisoformat(start_date))
    if end_date:
        stmt = stmt.where(AthleteStream.recorded_at <= datetime.fromisoformat(end_date))
    rows = db.scalars(stmt).all()
    return [
        Stream(
            metric_key=r.metric_key,
            stream_type=r.stream_type,
            data=r.data,
            axis=r.axis,
            axis_type=r.axis_type,
            original_size=r.original_size,
            resolution=r.resolution,
        )
        for r in rows
    ]


@router.get("/preferences", response_model=AthletePreferences)
def get_preferences(athlete: Athlete = Depends(get_current_athlete), db: Session = Depends(get_db)):
    return athlete_service.get_preferences(db, athlete.id)


@router.patch("/preferences", response_model=SuccessResponse)
def update_preferences(
    body: UpdatePreferencesRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    data = body.model_dump(exclude_unset=True)
    if body.privacy_settings:
        data["privacy_settings"] = body.privacy_settings.model_dump()
    if body.email_notifications:
        data["email_notifications"] = body.email_notifications.model_dump()
    success, error = athlete_service.update_preferences(db, athlete.id, data)
    return SuccessResponse(success=success, error_message=error)
