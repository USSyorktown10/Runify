from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.models.social import Club, ClubMember
from app.schemas.club import (
    CreateClubRequest,
    CreatePostRequest,
    DetailedClub,
    PaginatedClubsResponse,
    UpdateClubPreferencesRequest,
)
from app.schemas.common import SuccessResponse
from app.schemas.social import PaginatedAthletesResponse
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
def get_club(club_id: str, db: Session = Depends(get_db)):
    club = db.get(Club, club_id)
    if not club:
        from app.core.errors import NotFoundError
        raise NotFoundError()
    return club_service.to_detailed(db, club)


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
