from pydantic import BaseModel, Field

from app.schemas.common import (
    DynamicActivityZone,
    DynamicMetricDistribution,
    DynamicWorkoutMetric,
    PaginatedResponseMetadata,
)


class SummaryActivity(BaseModel):
    id: str
    athlete_id: str
    name: str
    activity_type: str
    distance: float
    moving_time: int
    start_date: str
    polyline_summary: str
    visibility: str
    biometrics_visibility: str
    like_count: int
    comment_count: int
    is_liked: bool
    metrics: list[DynamicWorkoutMetric]


class DetailedActivity(BaseModel):
    id: str
    athlete_id: str
    name: str
    description: str
    activity_type: str
    distance: float
    moving_time: int
    elapsed_time: int
    start_date: str
    polyline: str
    device_name: str
    gear_id: str | None
    perceived_exertion: int | None
    visibility: str
    biometrics_visibility: str
    like_count: int
    comment_count: int
    is_liked: bool
    metrics: list[DynamicWorkoutMetric]
    distributions: list[DynamicMetricDistribution]
    zones: list[DynamicActivityZone]
    laps: list["Lap"]


class Lap(BaseModel):
    id: str
    lap_index: int
    name: str
    start_date: str
    elapsed_time: int
    moving_time: int
    distance: float
    average_speed: float


class Split(BaseModel):
    index: int
    distance: float
    elapsed_time: int
    elevation_difference: float
    average_speed: float


class PowerCurveValue(BaseModel):
    time_interval_seconds: int
    power_value_watts: float


class PowerCurve(BaseModel):
    curve_values: list[PowerCurveValue]


class Stream(BaseModel):
    metric_key: str
    stream_type: str
    data: list[float]
    axis: list[float]
    axis_type: str
    original_size: int = 0
    resolution: str = "high"


class GetStreamRequest(BaseModel):
    streams: list[str] = Field(default_factory=list)
    resolution: str = "high"
    start_date: str | None = None
    end_date: str | None = None


class CreateActivityRequest(BaseModel):
    name: str
    activity_type: str = "run"
    start_date: str
    elapsed_time: int
    distance: float
    description: str = ""
    perceived_exertion: int | None = None
    gear_id: str | None = None
    visibility: str | None = None
    biometrics_visibility: str | None = None
    metrics: list[DynamicWorkoutMetric] = Field(default_factory=list)


class UpdateActivityRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    gear_id: str | None = None
    visibility: str | None = None
    biometrics_visibility: str | None = None


class CropActivityRequest(BaseModel):
    start_index: int
    end_index: int


class PaginatedActivitiesResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SummaryActivity]
