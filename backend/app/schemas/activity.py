from pydantic import BaseModel, Field

from backend.app.schemas.common import (
    ActivityZones,
    MetricDistribution,
    WorkoutMetric,
    PaginatedResponseMetadata,
)

#Activity displayed on feed
class SummaryActivity(BaseModel):
    id: str
    athlete_id: str
    name: str
    activity_type: str
    distance: float
    moving_time: float
    start_date: str
    polyline_summary: str
    device_name: str = ""
    visibility: str
    metrics_visibility: str
    like_count: int
    comment_count: int
    is_liked: bool
    metrics: list[WorkoutMetric]
    
#Activity when opened
class DetailedActivity(BaseModel):
    id: str
    athlete_id: str
    name: str
    activity_type: str
    distance: float
    moving_time: float
    elapsed_time: float
    start_date: str
    polyline: str
    device_name: str = ""
    average_speed: float
    average_heartrate: int
    kilojoules: float
    gear_id: str | None
    percived_exertion: int | None
    visibility: str
    metrics_visibility: str
    like_count: int
    comment_count: int
    is_liked: bool
    metrics: list[WorkoutMetric]
    distributions: list[MetricDistribution] 
    zones: list[ActivityZones]
    laps: list["Lap"]

#Stores data for each lap/mile
class Lap(BaseModel):
    id: str
    lap_index: int
    name: str
    start_date: str
    elapsed_time: int
    moving_time: int
    distance: float
    average_speed: float
    #metrics: list[WorkoutMetric] (Dont need because of GetStreamRequest)
    
#Any lap mark between 2 points
class Split(BaseModel):
    index: int
    distance: float
    elapsed_time: int
    elevation_difference: float
    average_speed: float

#Single curve value
class PowerCurveValue(BaseModel):
    time_interval_seconds: int
    power_value_watts: float

#Curve value list
class PowerCurve(BaseModel):
    curve_values: list[PowerCurveValue]

#Data point stream
class Stream(BaseModel):
    metric_key: str
    stream_type: str
    data: list[float]
    axis: list[float]
    axis_type: str
    original_size: int = 0
    resolution: str = "high"

#Get specific data from a stream of points
class GetStreamRequest(BaseModel):
    streams: list[str] = Field(default_factory=list)
    resolution: str = "high"
    start_date: str | None = None
    end_date: str | None = None

#Creates activity
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
    metrics_visibility: str | None = None
    metrics: list[WorkoutMetric] = Field(default_factory=list)

#Updates activity
class UpdateActivityRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    gear_id: str | None = None
    visibility: str | None = None
    metrics_visibility: str | None = None

#Crops activity
class CropActivityRequest(BaseModel):
    start_index: int
    end_index: int

#Makes a page so my browser dosent load all 73462847327487 of my activities
class PaginatedActivitiesResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[SummaryActivity]