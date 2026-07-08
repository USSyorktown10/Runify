from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete, get_optional_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.models.social import Club, ClubMember
from app.schemas.activity import PaginatedActivitiesResponse
from app.schemas.club import (
    CreateClubRequest,
    CreatePostRequest,
    DetailedClub,
    PaginatedClubLeaderboardResponse,
    PaginatedClubsResponse,
    UpdateClubPreferencesRequest,
)
from app.schemas.common import SuccessResponse
from app.schemas.social import PaginatedAthletesResponse, PaginatedPostsResponse
from app.services.athlete_service import to_summary
from app.services.club_service import club_service

router = APIRouter(tags=["clubs"])


@router.post("/clubs")
def create_club(
    body: CreateClubRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    club = club_service.create(db, athlete.id, body.model_dump())
    return {"club": club, "success": True}


@router.get("/clubs", response_model=PaginatedClubsResponse)
def search_clubs(
    query: str | None = Query(None),
    tag: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = select(Club)
    if query:
        stmt = stmt.where(Club.name.ilike(f"%{query}%"))
    items, pagination = paginate_offset(db, stmt.order_by(Club.created_at.desc()), page, per_page)
    return PaginatedClubsResponse(
        pagination=pagination,
        items=[club_service.to_summary(c) for c in items],
    )


@router.get("/athletes/{athlete_id}/clubs", response_model=PaginatedClubsResponse)
def athlete_clubs(
    athlete_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = select(Club).join(ClubMember, ClubMember.club_id == Club.id).where(ClubMember.athlete_id == athlete_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedClubsResponse(pagination=pagination, items=[club_service.to_summary(c) for c in items])


@router.get("/clubs/{club_id}", response_model=DetailedClub)
def get_club(
    club_id: str,
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    club = db.get(Club, club_id)
    if not club:
        from app.core.errors import NotFoundError
        raise NotFoundError()
    viewer_id = viewer.id if viewer else None
    return club_service.to_detailed(db, club, viewer_id)


@router.get("/clubs/{club_id}/members", response_model=PaginatedAthletesResponse)
def club_members(
    club_id: str,
    query: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = select(Athlete).join(ClubMember, ClubMember.athlete_id == Athlete.id).where(ClubMember.club_id == club_id)
    if query:
        stmt = stmt.where(Athlete.username.ilike(f"%{query}%"))
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.get("/clubs/{club_id}/posts", response_model=PaginatedPostsResponse)
def club_posts(
    club_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    viewer_id = viewer.id if viewer else None
    items, pagination = club_service.list_posts(db, club_id, viewer_id, page, per_page)
    return PaginatedPostsResponse(pagination=pagination, items=items)


@router.get("/clubs/{club_id}/leaderboard", response_model=PaginatedClubLeaderboardResponse)
def club_leaderboard(
    club_id: str,
    period: str = Query("this_week"),
    metric: str = Query("distance"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    viewer_id = viewer.id if viewer else None
    items, pagination, viewer_summary = club_service.leaderboard(
        db, club_id, viewer_id, period, metric, page, per_page
    )
    return PaginatedClubLeaderboardResponse(
        pagination=pagination, items=items, viewer_summary=viewer_summary
    )


@router.get("/clubs/{club_id}/recent-activity", response_model=PaginatedActivitiesResponse)
def club_recent_activity(
    club_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    viewer: Athlete | None = Depends(get_optional_athlete),
    db: Session = Depends(get_db),
):
    viewer_id = viewer.id if viewer else None
    items, pagination = club_service.recent_activity(db, club_id, viewer_id, page, per_page)
    return PaginatedActivitiesResponse(
        pagination=pagination,
        items=items,
    )


@router.get("/clubs/{club_id}/join-requests", response_model=PaginatedAthletesResponse)
def club_join_requests(
    club_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    items, pagination = club_service.list_join_requests(db, club_id, athlete.id, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/clubs/{club_id}/join-requests/{athlete_id}/accept", response_model=SuccessResponse)
def accept_join_request(
    club_id: str,
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.accept_join_request(db, club_id, athlete_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/clubs/{club_id}/join-requests/{athlete_id}/deny", response_model=SuccessResponse)
def deny_join_request(
    club_id: str,
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.deny_join_request(db, club_id, athlete_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/clubs/{club_id}/invites/accept", response_model=SuccessResponse)
def accept_club_invite(
    club_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.accept_invite(db, club_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/clubs/{club_id}/invites/deny", response_model=SuccessResponse)
def deny_club_invite(
    club_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.deny_invite(db, club_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/clubs/{club_id}/members/{user_id}", response_model=SuccessResponse)
def remove_club_member(
    club_id: str,
    user_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.remove_member(db, club_id, user_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/clubs/{club_id}", response_model=SuccessResponse)
def delete_club(
    club_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    club = db.get(Club, club_id)
    if not club or club.creator_id != athlete.id:
        return SuccessResponse(success=False, error_message="Not found or forbidden")
    db.delete(club)
    db.commit()
    return SuccessResponse(success=True)


@router.patch("/clubs/{club_id}/invite", response_model=SuccessResponse)
def invite_to_club(
    club_id: str,
    athlete_id: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.invite(db, club_id, athlete.id, athlete_id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/clubs/{club_id}/join", response_model=SuccessResponse)
def join_club(
    club_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = club_service.join(db, club_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/clubs/{club_id}/posts")
def create_club_post(
    club_id: str,
    body: CreatePostRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    post = club_service.create_post(db, club_id, athlete.id, body.title, body.body)
    return {"post": post, "success": True, "error_message": None}


@router.patch("/clubs/{club_id}/preferences", response_model=SuccessResponse)
def update_club(
    club_id: str,
    body: UpdateClubPreferencesRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    club = db.get(Club, club_id)
    if not club or not club_service._is_admin(db, club_id, athlete.id):
        return SuccessResponse(success=False, error_message="Forbidden")
    for field in ("name", "description", "profile_picture_url", "cover_photo_url", "is_private", "tags"):
        val = getattr(body, field, None)
        if val is not None:
            setattr(club, field, val)
    db.commit()
    return SuccessResponse(success=True)
