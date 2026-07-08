from pydantic import BaseModel

from app.schemas.common import (
    PaginatedResponseMetadata,
    PrivacySettings,
)


class SummaryAthlete(BaseModel):
    id: str
    first_name: str
    last_name: str
    profile_picture_url: str
    city: str = ""
    state: str = ""
    country: str = ""


class MeAthlete(SummaryAthlete):
    username: str
    email: str
    city: str
    state: str
    country: str
    gender: str | None = None
    birthdate: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    created: str
    wearable_connected: bool
    privacy_settings: PrivacySettings


class AthleteStats(BaseModel):
    current_ftp: int
    threshold_pace: float
    ytd_run_totals: float
    all_time_run_totals: float


class PersonalRecord(BaseModel):
    distance_name: str
    time_in_seconds: int
    activity_id: str
    achieved_date: str


class DetailedAthlete(BaseModel):
    id: str
    username: str
    first_name: str
    last_name: str
    city: str
    state: str
    country: str
    profile_picture_url: str
    created: str
    wearable_connected: bool
    stats: AthleteStats
    privacy_settings: PrivacySettings
    personal_records: list[PersonalRecord]


class ConnectionSearchResult(BaseModel):
    athlete: SummaryAthlete
    relationship_status: str
    connection_degree: int
    common_clubs_count: int
    mutual_followers_count: int


class PaginatedConnectionsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[ConnectionSearchResult]


class UpdateAthleteStatsRequest(BaseModel):
    current_ftp: int | None = None
    threshold_pace: float | None = None


class UpdateAthleteProfileRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    profile_picture_url: str | None = None
    gender: str | None = None
    birthdate: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None


class EmailNotificationsSettings(BaseModel):
    comments: bool = True
    likes: bool = True
    follow_requests: bool = True
    club_invites: bool = True


class AthletePreferences(BaseModel):
    measurement_system: str
    privacy_settings: PrivacySettings
    theme: str
    email_notifications: EmailNotificationsSettings


class UpdatePreferencesRequest(BaseModel):
    measurement_system: str | None = None
    privacy_settings: PrivacySettings | None = None
    theme: str | None = None
    email_notifications: EmailNotificationsSettings | None = None
