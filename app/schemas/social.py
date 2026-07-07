from pydantic import BaseModel

from app.schemas.activity import SummaryActivity
from app.schemas.athlete import SummaryAthlete
from app.schemas.common import PaginatedResponseMetadata


class Comment(BaseModel):
    id: str
    author: SummaryAthlete
    text: str
    created_at: str
    like_count: int
    is_liked: bool


class AthletePost(BaseModel):
    id: str
    athlete_id: str
    text: str
    media_urls: list[str]
    created_at: str
    like_count: int
    comment_count: int
    is_liked: bool


class Post(BaseModel):
    id: str
    club_id: str
    author: SummaryAthlete
    title: str
    body: str
    created_at: str


class CreateAthletePostRequest(BaseModel):
    text: str
    media_urls: list[str] | None = None


class UpdateAthletePostRequest(BaseModel):
    text: str | None = None
    media_urls: list[str] | None = None


class FeedItem(BaseModel):
    id: str
    type: str
    athlete: SummaryAthlete
    created_at: str
    activity_data: SummaryActivity | None = None
    post_data: AthletePost | None = None
    club_post_data: Post | None = None


class CursorPaginatedFeedResponse(BaseModel):
    next_cursor: str | None
    items: list[FeedItem]


class PaginatedCommentsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[Comment]


class PaginatedAthletesResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SummaryAthlete]
