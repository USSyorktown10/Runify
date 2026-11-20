from pydantic import BaseModel, EmailStr, Field
from datetime import date

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    session_token: str | None = None
    error_message: str | None = None

class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(min_length=8)
    gender: str | None = None
    birthdate: date | None = None
    weight_kg: float | None = Field(default=None, gt=0)
    height_cm: float | None = Field(default=None, gt=0)

class SignupResponse(BaseModel):
    success: bool
    error_message: str | None = None