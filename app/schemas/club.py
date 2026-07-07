from pydantic import BaseModel

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
