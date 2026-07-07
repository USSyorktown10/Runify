from pydantic import BaseModel

from app.schemas.common import PaginatedResponseMetadata


class SummarySegment(BaseModel):
    id: str
    name: str
    activity_type: str
    distance: float
    average_grade: float
    start_latlng: list[float]
    end_latlng: list[float]
    is_starred: bool


class DetailedSegment(BaseModel):
    id: str
    name: str
    activity_type: str
    distance: float
    average_grade: float
    start_latlng: list[float]
    end_latlng: list[float]
    is_starred: bool
    polyline: str
    elevation_high: float
    elevation_low: float
    total_effort_count: int
    total_athlete_count: int
    star_count: int


class SegmentEffort(BaseModel):
    id: str
    segment_id: str
    activity_id: str
    elapsed_time: int
    moving_time: int
    start_date: str
    average_heartrate: float | None
    average_power: float | None
    rank: int | None = None


class LeaderboardEntry(BaseModel):
    athlete_id: str
    athlete_name: str
    athlete_profile_picture_url: str
    rank: int
    elapsed_time: int
    average_hr: float | None
    average_power: float | None
    achieved_date: str


class PaginatedSegmentsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SummarySegment]


class PaginatedSegmentEffortsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SegmentEffort]


class PaginatedLeaderboardResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[LeaderboardEntry]
