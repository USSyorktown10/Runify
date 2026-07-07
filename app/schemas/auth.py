
from pydantic import BaseModel, EmailStr

from app.schemas.common import ClientMetadata, UserMetadata


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    session_token: str | None = None
    error_message: str | None = None


class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    metadata: UserMetadata | None = None


class SignupResponse(BaseModel):
    success: bool
    error_message: str | None = None


class LogoutRequest(BaseModel):
    session_token: str


class RefreshRequest(BaseModel):
    session_token: str


class RefreshResponse(BaseModel):
    success: bool
    session_token: str | None = None
    expiration_time: str | None = None


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str


class VerifyEmailRequest(BaseModel):
    signup_token: str


class SSORequest(BaseModel):
    oauth_token: str
    client_metadata: ClientMetadata | None = None


class SSOResponse(BaseModel):
    session_token: str | None = None
    success: bool = False
    error_message: str | None = None


class ActiveSession(BaseModel):
    session_id: str
    client_metadata: ClientMetadata
    ip_address: str
    location: str | None = None
    last_active_at: str
    created_at: str
    is_current: bool
