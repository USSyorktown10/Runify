from pydantic import BaseModel, ConfigDict, EmailStr, Field
from app.schemas.common import BaseSchema

class SummaryUser(BaseModel):
    id: int
    username: str
    first_name: str
    last_name: str
    city: str
    state: str
    country: str
    profile_picture_url: str | None
    

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