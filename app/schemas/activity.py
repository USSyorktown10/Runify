from pydantic import BaseModel, Field
from app.schemas.common import (
    BaseSchema, 
    ActivityMetric, 
    ActivityDistribution,
    ActivityZone
)

class SummaryActivity(BaseModel):
    id: str
    user_id: str
    name: str
    activity_type: str
    distance: float #km
    moving_time: int
    start_date: str
    polyline_summary: str | None = None
    device_name: str = ""
    visibility: str
    metrics_visibility: str
    like_count: int
    comment_count: int
    is_liked: bool
    metrics: list[ActivityMetric]
    
class DetailedActivity(BaseModel):
    id: str
    user_id: str
    name: str
    description: str
    activity_type: str
    distance: float
    moving_time: int
    elapsed_time: int
    start_date: str
    polyline: str | None = None
    device_name: str
    gear_id: str | None
    percieved_effort: int | None
    visibility: str
    metrics_visibility: str
    like_count: int
    comment_count: int
    is_liked: bool
    metrics: list[ActivityMetric]
    distributions: list[ActivityDistribution]
    zones: list[ActivityZone]
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
    metrics: list[ActivityMetric]
    
class PowerCurveValues(BaseModel):
    time_interval_seconds: int
    power_watts: float
    
class PowerCurve(BaseModel):
    curve_values: list[PowerCurveValues]
    
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
    percieved_effort: int | None = None
    gear_id: str | None = None
    visibility: str | None = None
    metrics_visibility: str | None = None
    metrics: list[ActivityMetric] = Field(default_factory=list)
    
class UpdateActivityRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    gear_id: str | None = None
    visibility: str | None = None
    biometrics_visibility: str | None = None
    
class CropActivityRequest(BaseModel):
    start_index: int
    end_index: int