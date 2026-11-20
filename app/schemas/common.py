from pydantic import BaseModel, ConfigDict, Field

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class ActivityMetric(BaseModel):
    key: str
    value: float
    source: str
    unit: str
    display_name: str

class DistributionBucket(BaseModel):
    min_value: float
    max_value: float
    time_seconds: int

class ActivityDistribution(BaseModel):
    key: str
    display_name: str
    unit: str
    buckets: list[DistributionBucket]

class ZoneData(BaseModel):
    zone_index: int
    min_value: float
    max_value: float
    time_seconds: int

class ActivityZone(BaseModel):
    key: str
    display_name: str
    unit: str
    reference_value: float | None = None
    reference_name: str | None = None
    zones: list[ZoneData]

class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    skip: int
    limit: int

class Message(BaseModel):
    detail: str