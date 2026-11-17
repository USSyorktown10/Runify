from pydantic import BaseModel, ConfigDict, EmailStr, Field
from app.schemas.common import BaseSchema, SummaryStats, PersonalRecord, PrivacySettings

class SummaryUser(BaseModel):
    id: int
    username: str
    first_name: str
    last_name: str
    city: str = ""
    state: str = ""
    country: str = ""
    profile_picture_url: str | None
    
class DetailedUser(BaseModel):
    id: int
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
    stats: SummaryStats
    personal_records: list[PersonalRecord]
    privacy_settings: PrivacySettings
    
# User editing, creaation, and reponse (fix more later)
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=8)
    
class UserUpdate(BaseModel):
    username: str | None = Field(default=None, min_length=3, max_length=30)
    password: str | None = Field(default=None, min_length=8)
    description: str | None = Field(default=None, max_length=250)

class UserResponse(BaseSchema):
    id: int
    email: EmailStr