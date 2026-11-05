from pydantic import BaseModel, Field

class PaginatedResponseMetadata(BaseModel):
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)
    sort_order: str = "newest"

class WorkoutMetric(BaseModel):
    key: str
    value: float
    source: str
    unit: str
    display_name: str

class ActivityZones(BaseModel):
    key: str
    display_name: str
    unit: str
    reference_value: float | None = None
    reference_name: str | None = None
    zones: list[ZoneData]
    
class ZoneData(BaseModel):
    zone_index: int
    min_value: float
    max_value: float
    time_in_seconds: int
    
class MetricDistribution(BaseModel):
    key: str
    display_name: str
    unit: str
    reference_value: float | None = None
    reference_name: str | None = None
    zones: list[ZoneData]