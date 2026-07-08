from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.models.social import Comment as CommentModel, Like as LikeModel
from app.schemas.common import SuccessResponse
from app.schemas.social import (
    Comment as CommentSchema,
)
from app.schemas.social import (
    CreateAthletePostRequest,
    CursorPaginatedFeedResponse,
    PaginatedAthletesResponse,
    PaginatedCommentsResponse,
)
from app.services.athlete_service import to_summary
from app.services.feed_service import feed_service
from app.services.social_service import social_service

router = APIRouter(tags=["social"])


@router.get("/activities/{activity_id}/comments", response_model=PaginatedCommentsResponse)
def activity_comments(
    activity_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_comments(db, "activity", activity_id)
    return _comments_response(db, athlete, stmt, page, per_page)


@router.post("/activities/{activity_id}/comments")
def post_activity_comment(
    activity_id: str,
    text: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = social_service.add_comment(db, athlete.id, "activity", activity_id, text)
    return {"comment": comment, "success": True}


@router.patch("/activities/{activity_id}/comments/{comment_id}", response_model=SuccessResponse)
def edit_activity_comment(
    activity_id: str,
    comment_id: str,
    text: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = db.get(CommentModel, comment_id)
    if not comment or comment.author_id != athlete.id:
        return SuccessResponse(success=False, error_message="Comment not found")
    comment.text = text
    db.commit()
    return SuccessResponse(success=True)


@router.delete("/activities/{activity_id}/comments/{comment_id}", response_model=SuccessResponse)
def delete_activity_comment(
    activity_id: str,
    comment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = db.get(CommentModel, comment_id)
    if not comment or comment.author_id != athlete.id:
        return SuccessResponse(success=False, error_message="Comment not found")
    db.delete(comment)
    db.commit()
    return SuccessResponse(success=True)


@router.get("/activities/{activity_id}/likes", response_model=PaginatedAthletesResponse)
def activity_likers(
    activity_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_likers(db, "activity", activity_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/activities/{activity_id}/likes", response_model=SuccessResponse)
def like_activity(
    activity_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.like(db, athlete.id, "activity", activity_id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/activities/{activity_id}/likes", response_model=SuccessResponse)
def unlike_activity(
    activity_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.unlike(db, athlete.id, "activity", activity_id)
    return SuccessResponse(success=success, error_message=error)


@router.get("/posts/{post_id}/likes", response_model=PaginatedAthletesResponse)
def post_likers(
    post_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_likers(db, "post", post_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/posts/{post_id}/likes", response_model=SuccessResponse)
def like_post(
    post_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.like(db, athlete.id, "post", post_id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/posts/{post_id}/likes", response_model=SuccessResponse)
def unlike_post(
    post_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.unlike(db, athlete.id, "post", post_id)
    return SuccessResponse(success=success, error_message=error)


def _comments_response(db: Session, athlete: Athlete, stmt, page: int, per_page: int) -> PaginatedCommentsResponse:
    items, pagination = paginate_offset(db, stmt, page, per_page)
    comment_ids = [c.id for c in items]
    liked_set: set[str] = set()
    like_counts: dict[str, int] = {}
    if comment_ids:
        liked_ids = db.scalars(
            select(LikeModel.target_id).where(
                LikeModel.athlete_id == athlete.id,
                LikeModel.target_type == "comment",
                LikeModel.target_id.in_(comment_ids),
            )
        ).all()
        liked_set = set(liked_ids)
        like_counts = social_service.comment_like_counts(db, comment_ids)
    result = []
    for c in items:
        author = db.get(Athlete, c.author_id)
        result.append(
            CommentSchema(
                id=c.id,
                author=to_summary(author),
                text=c.text,
                created_at=c.created_at.isoformat(),
                like_count=like_counts.get(c.id, 0),
                is_liked=c.id in liked_set,
            )
        )
    return PaginatedCommentsResponse(pagination=pagination, items=result)


@router.get("/posts/{post_id}/comments", response_model=PaginatedCommentsResponse)
def post_comments(
    post_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_comments(db, "post", post_id)
    return _comments_response(db, athlete, stmt, page, per_page)


@router.post("/posts/{post_id}/comments")
def create_post_comment(
    post_id: str,
    text: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = social_service.add_comment(db, athlete.id, "post", post_id, text)
    return {"comment": comment, "success": True}


@router.patch("/posts/{post_id}/comments/{comment_id}", response_model=SuccessResponse)
def edit_post_comment(
    post_id: str,
    comment_id: str,
    text: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = db.get(CommentModel, comment_id)
    if not comment or comment.author_id != athlete.id or comment.target_id != post_id:
        return SuccessResponse(success=False, error_message="Comment not found")
    comment.text = text
    db.commit()
    return SuccessResponse(success=True)


@router.delete("/posts/{post_id}/comments/{comment_id}", response_model=SuccessResponse)
def delete_post_comment(
    post_id: str,
    comment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = db.get(CommentModel, comment_id)
    if not comment or comment.author_id != athlete.id:
        return SuccessResponse(success=False, error_message="Comment not found")
    db.delete(comment)
    db.commit()
    return SuccessResponse(success=True)


@router.get("/club-posts/{post_id}/likes", response_model=PaginatedAthletesResponse)
def club_post_likers(
    post_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_likers(db, "club_post", post_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/club-posts/{post_id}/likes", response_model=SuccessResponse)
def like_club_post(
    post_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.like(db, athlete.id, "club_post", post_id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/club-posts/{post_id}/likes", response_model=SuccessResponse)
def unlike_club_post(
    post_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.unlike(db, athlete.id, "club_post", post_id)
    return SuccessResponse(success=success, error_message=error)


@router.get("/club-posts/{post_id}/comments", response_model=PaginatedCommentsResponse)
def club_post_comments(
    post_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_comments(db, "club_post", post_id)
    return _comments_response(db, athlete, stmt, page, per_page)


@router.post("/club-posts/{post_id}/comments")
def create_club_post_comment(
    post_id: str,
    text: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = social_service.add_comment(db, athlete.id, "club_post", post_id, text)
    return {"comment": comment, "success": True}


@router.patch("/club-posts/{post_id}/comments/{comment_id}", response_model=SuccessResponse)
def edit_club_post_comment(
    post_id: str,
    comment_id: str,
    text: str = Query(...),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = db.get(CommentModel, comment_id)
    if not comment or comment.author_id != athlete.id or comment.target_id != post_id:
        return SuccessResponse(success=False, error_message="Comment not found")
    comment.text = text
    db.commit()
    return SuccessResponse(success=True)


@router.delete("/club-posts/{post_id}/comments/{comment_id}", response_model=SuccessResponse)
def delete_club_post_comment(
    post_id: str,
    comment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    comment = db.get(CommentModel, comment_id)
    if not comment or comment.author_id != athlete.id:
        return SuccessResponse(success=False, error_message="Comment not found")
    db.delete(comment)
    db.commit()
    return SuccessResponse(success=True)


@router.get("/athlete/feed", response_model=CursorPaginatedFeedResponse)
def home_feed(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    items, next_cursor = feed_service.get_home_feed(db, athlete, cursor, limit)
    return CursorPaginatedFeedResponse(next_cursor=next_cursor, items=items)


@router.get("/athletes/{athlete_id}/feed", response_model=CursorPaginatedFeedResponse)
def athlete_feed(
    athlete_id: str,
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    viewer: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    items, next_cursor = feed_service.get_athlete_profile_feed(db, athlete_id, cursor, limit, viewer.id)
    return CursorPaginatedFeedResponse(next_cursor=next_cursor, items=items)


@router.post("/athletes/{athlete_id}/follow")
def follow_athlete(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    status, success = social_service.follow(db, athlete.id, athlete_id)
    return {"status": status, "success": success}


@router.delete("/athletes/{athlete_id}/follow", response_model=SuccessResponse)
def unfollow_athlete(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.unfollow(db, athlete.id, athlete_id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/athletes/{athlete_id}/follow/accept", response_model=SuccessResponse)
def accept_follow(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.accept_follow(db, athlete.id, athlete_id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/athletes/{athlete_id}/follow/deny", response_model=SuccessResponse)
def deny_follow(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.deny_follow(db, athlete.id, athlete_id)
    return SuccessResponse(success=success, error_message=error)


@router.get("/athlete/follow-requests", response_model=PaginatedAthletesResponse)
def follow_requests(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_follow_requests(db, athlete.id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.get("/athletes/{athlete_id}/followers", response_model=PaginatedAthletesResponse)
def followers(
    athlete_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_followers(db, athlete_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.get("/athletes/{athlete_id}/following", response_model=PaginatedAthletesResponse)
def following(
    athlete_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_following(db, athlete_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/athletes/{athlete_id}/posts")
def create_post(
    athlete_id: str,
    body: CreateAthletePostRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    if athlete.id != athlete_id:
        from app.core.errors import ForbiddenError
        raise ForbiddenError()
    post = social_service.create_post(db, athlete.id, body.text, body.media_urls)
    return {"post": post, "success": True}


@router.get("/athletes/{athlete_id}/relationship")
def relationship(
    athlete_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    return {"status": social_service.relationship_status(db, athlete.id, athlete_id)}


@router.get("/comments/{comment_id}/likes", response_model=PaginatedAthletesResponse)
def comment_likers(
    comment_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    stmt = social_service.list_likers(db, "comment", comment_id)
    items, pagination = paginate_offset(db, stmt, page, per_page)
    return PaginatedAthletesResponse(pagination=pagination, items=[to_summary(a) for a in items])


@router.post("/comments/{comment_id}/likes", response_model=SuccessResponse)
def like_comment(
    comment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.like(db, athlete.id, "comment", comment_id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/comments/{comment_id}/likes", response_model=SuccessResponse)
def unlike_comment(
    comment_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = social_service.unlike(db, athlete.id, "comment", comment_id)
    return SuccessResponse(success=success, error_message=error)
