from pydantic import BaseModel, ConfigDict, EmailStr, Field
from app.schemas.common import BaseSchema

#Main
class SummaryUser(BaseModel):
    id: str
    username: str
    first_name: str
    last_name: str
    city: str = ""
    state: str = ""
    country: str = ""
    profile_picture_url: str | None
    
class DetailedUser(BaseModel):
    id: str
    username: str
    first_name: str
    last_name: str
    city: str
    state: str
    country: str
    profile_picture_url: str | None
    bio: str | None = Field(default=None, max_length=200)
    created: str
    primary_wearable: str
    wearable_connected: bool
    stats: SummaryStats
    personal_records: list[PersonalRecord]
    privacy_settings: PrivacySettings
    
class MeUser(SummaryUser):
    username: str
    email: str
    bio: str | None = Field(default=None, max_length=200)
    gender: str | None = None
    birthdate: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    created: str
    primary_wearable: str
    wearable_connected: bool
    privacy_settings: PrivacySettings
    
#Helpers
class SummaryStats(BaseModel):
    current_ftp: int
    threshold_pace: float
    four_week_run_totals: list[float] = Field(default_factory=list, min_length=4, max_length=4) #weekly mileage for 4 weeks
    ytd_run_totals: float
    all_time_run_totals: float
    
class PersonalRecord(BaseModel):
    distance_name: str
    distance_meters: str
    time: float #seconds
    activity_id: int
    date: str
    
class PrivacySettings(BaseModel): #public, followers, marked friends, no one
    profile_visibility: str = "public"
    activity_visibility: str = "followers"
    metrics_visibility: str = "followers"
    map_visibility: str = "followers"
    request_to_follow: bool = True

class EmailNotificationsSettings(BaseModel):
    comments: bool = True
    likes: bool = True
    follow_requests: bool = True
    club_invites: bool = True

class UserPreferences(BaseModel):
    measurement_system: str
    privacy_settings: PrivacySettings
    theme: str
    email_notifications: EmailNotificationsSettings

#Updates
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=8)
    
class UpdateUserProfileRequest(BaseModel):
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    profile_picture_url: str | None = None
    birthdate: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    bio: str | None = None

class UpdatePersonalRecordRequest(BaseModel):
    distance_name: str | None = None
    time: float | None = None
    activity_id: int | None = None
    date: str | None = None
    
class UpdateUserPreferencesRequest(BaseModel):
    measurement_system: str | None = None
    privacy_settings: PrivacySettings | None = None
    theme: str | None = None
    email_notifications: EmailNotificationsSettings | None = None
    
class UpdateUserStatsRequest(BaseModel):
    current_ftp: int | None = None
    threshold_pace: float | None = None
    four_week_run_totals: list[float] | None = None
    ytd_run_totals: float
    all_time_run_totals: float | None = None

class UserResponse(BaseSchema):
    id: str
    email: EmailStr