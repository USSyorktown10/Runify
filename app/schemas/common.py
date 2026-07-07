from pydantic import BaseModel, Field


class PaginatedRequest(BaseModel):
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)
    sort_order: str = "newest"


class CursorPaginationRequest(BaseModel):
    cursor: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    sort_order: str | None = "newest"


class PaginatedResponseMetadata(BaseModel):
    page: int
    per_page: int
    total_items: int
    total_pages: int


class SuccessResponse(BaseModel):
    success: bool = True
    error_message: str | None = None


class DistributionBucket(BaseModel):
    min_value: float
    max_value: float
    time_in_seconds: int


class ZoneData(BaseModel):
    zone_index: int
    min_value: float
    max_value: float
    time_in_seconds: int


class PrivacySettings(BaseModel):
    profile_visibility: str = "public"
    activity_visibility: str = "followers"
    biometrics_visibility: str = "followers"


class DynamicWorkoutMetric(BaseModel):
    key: str
    value: float
    source: str
    unit: str
    display_name: str


class DynamicMetricDistribution(BaseModel):
    key: str
    display_name: str
    unit: str
    buckets: list[DistributionBucket]


class DynamicActivityZone(BaseModel):
    key: str
    display_name: str
    unit: str
    reference_value: float | None = None
    reference_name: str | None = None
    zones: list[ZoneData]


class ClientMetadata(BaseModel):
    user_agent: str = ""
    browser_name: str = ""
    browser_version: str = ""
    os_name: str = ""


class UserMetadata(BaseModel):
    gender: str | None = None
    birthdate: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
