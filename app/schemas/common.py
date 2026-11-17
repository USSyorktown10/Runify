from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    skip: int
    limit: int

class Message(BaseModel):
    detail: str
    
class SummaryStats(BaseModel):
    current_ftp: int
    threshold_pace: float
    four_week_run_totals: list[float]
    ytd_run_totals: float
    all_time_run_totals: float
    
class PersonalRecord(BaseModel):
    distance_name: str
    time: float #seconds
    activity_id: int
    date: str
    
#public, followers, marked friends, no one
class PrivacySettings(BaseModel):
    profile_visibility: str = "public"
    activity_visibility: str = "followers"
    metrics_visibility: str = "followers"
    map_visibility: str = "followers"