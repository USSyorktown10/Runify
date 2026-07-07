from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.models.segment import Segment, SegmentEffort, SegmentStar
from app.schemas.common import SuccessResponse
from app.schemas.segment import (
    DetailedSegment,
    LeaderboardEntry,
    PaginatedLeaderboardResponse,
    PaginatedSegmentEffortsResponse,
    PaginatedSegmentsResponse,
)
from app.schemas.segment import (
    SegmentEffort as SegmentEffortSchema,
)
from app.services.segment_service import segment_service

router = APIRouter(tags=["segments"])


@router.post("/segments")
def create_segment(
    activity_id: str = Query(...),
    start_index: int = Query(...),
    end_index: int = Query(...),
    name: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    seg = segment_service.create_from_activity(db, athlete.id, activity_id, start_index, end_index, name)
    return {"segment": seg, "success": True}


@router.get("/segments", response_model=PaginatedSegmentsResponse)
def search_segments(
    query: str | None = Query(None),
    activity_type: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = segment_service.search(db, query, activity_type)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedSegmentsResponse(
        pagination=pagination,
        items=[segment_service.to_summary(db, s, athlete.id) for s in items],
    )


@router.get("/athletes/{athlete_id}/segments", response_model=PaginatedSegmentsResponse)
def athlete_segments(
    athlete_id: str,
    starred_only: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    viewer: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    if starred_only:
        stmt = (
            select(Segment)
            .join(SegmentStar, SegmentStar.segment_id == Segment.id)
            .where(SegmentStar.athlete_id == athlete_id)
        )
    else:
        stmt = select(Segment).where(Segment.creator_id == athlete_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedSegmentsResponse(
        pagination=pagination,
        items=[segment_service.to_summary(db, s, viewer.id) for s in items],
    )


@router.get("/segments/{segment_id}", response_model=DetailedSegment)
def get_segment(
    segment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    seg = db.get(Segment, segment_id)
    if not seg:
        from app.core.errors import NotFoundError
        raise NotFoundError()
    return segment_service.to_detailed(db, seg, athlete.id)


@router.get("/segments/{segment_id}/efforts", response_model=PaginatedSegmentEffortsResponse)
def segment_efforts(
    segment_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = (
        select(SegmentEffort)
        .where(SegmentEffort.segment_id == segment_id, SegmentEffort.athlete_id == athlete.id)
        .order_by(SegmentEffort.start_date.desc())
    )
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedSegmentEffortsResponse(
        pagination=pagination,
        items=[
            SegmentEffortSchema(
                id=e.id,
                segment_id=e.segment_id,
                activity_id=e.activity_id,
                elapsed_time=e.elapsed_time,
                moving_time=e.moving_time,
                start_date=e.start_date.isoformat(),
                average_heartrate=e.average_heartrate,
                average_power=e.average_power,
            )
            for e in items
        ],
    )


@router.get("/segments/{segment_id}/leaderboard", response_model=PaginatedLeaderboardResponse)
def segment_leaderboard(
    segment_id: str,
    gender: str | None = Query(None),
    age_group: str | None = Query(None),
    weight_class: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    filtered = segment_service.leaderboard(db, segment_id, gender, age_group, weight_class, page, per_page)
    start = (page - 1) * per_page
    page_items = filtered[start : start + per_page]
    entries = []
    for rank, (effort, athlete) in enumerate(page_items, start + 1):
        entries.append(
            LeaderboardEntry(
                athlete_id=athlete.id,
                athlete_name=f"{athlete.first_name} {athlete.last_name}".strip(),
                athlete_profile_picture_url=athlete.profile_picture_url,
                rank=rank,
                elapsed_time=effort.elapsed_time,
                average_hr=effort.average_heartrate,
                average_power=effort.average_power,
                achieved_date=effort.start_date.isoformat(),
            )
        )
    from app.schemas.common import PaginatedResponseMetadata
    total = len(filtered)
    return PaginatedLeaderboardResponse(
        pagination=PaginatedResponseMetadata(
            page=page, per_page=per_page, total_items=total, total_pages=max(1, (total + per_page - 1) // per_page)
        ),
        items=entries,
    )


@router.post("/segments/{segment_id}/star", response_model=SuccessResponse)
def star_segment(
    segment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = segment_service.star(db, segment_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/segments/{segment_id}/star", response_model=SuccessResponse)
def unstar_segment(
    segment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = segment_service.unstar(db, segment_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)
