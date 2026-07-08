from pydantic import BaseModel

from app.schemas.athlete import SummaryAthlete
from app.schemas.common import PaginatedResponseMetadata


class SummaryClub(BaseModel):
    id: str
    name: str
    profile_picture_url: str
    member_count: int
    is_private: bool


class DetailedClub(BaseModel):
    id: str
    name: str
    description: str
    profile_picture_url: str
    cover_photo_url: str
    member_count: int
    is_private: bool
    creator_id: str
    created_at: str
    admins: list[str]
    tags: list[str]
    is_member: bool = False
    viewer_role: str | None = None
    has_pending_join_request: bool = False
    has_pending_invite: bool = False


class ClubLeaderboardEntry(BaseModel):
    rank: int
    athlete_id: str
    athlete: SummaryAthlete
    distance: float
    activity_count: int
    longest_activity_id: str | None
    longest_distance: float
    avg_pace: float | None
    elevation_gain: float


class ClubLeaderboardSummary(BaseModel):
    rank: int | None
    distance: float
    activity_count: int
    longest_distance: float
    avg_pace: float | None
    elevation_gain: float


class PaginatedClubLeaderboardResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[ClubLeaderboardEntry]
    viewer_summary: ClubLeaderboardSummary | None = None


class CreateClubRequest(BaseModel):
    name: str
    description: str = ""
    is_private: bool = False
    tags: list[str] = []


class UpdateClubPreferencesRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    profile_picture_url: str | None = None
    cover_photo_url: str | None = None
    is_private: bool | None = None
    tags: list[str] | None = None


class CreatePostRequest(BaseModel):
    title: str
    body: str


class UpdateClubPostRequest(BaseModel):
    title: str | None = None
    body: str | None = None


class PaginatedClubsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SummaryClub]
