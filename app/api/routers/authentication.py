from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.security import get_current_athlete, get_current_session
from app.db.session import get_db
from app.models.athlete import Athlete
from app.models.auth import AuthSession
from app.schemas.auth import (
    ActiveSession,
    ForgotPasswordRequest,
    LoginRequest,
    LoginResponse,
    LogoutRequest,
    RefreshRequest,
    RefreshResponse,
    ResetPasswordRequest,
    SignupRequest,
    SignupResponse,
    SSORequest,
    SSOResponse,
    VerifyEmailRequest,
)
from app.schemas.common import ClientMetadata, SuccessResponse
from app.services.auth_service import auth_service

router = APIRouter(prefix="/authentication", tags=["authentication"])


@router.post("/signup", response_model=SignupResponse)
def signup(body: SignupRequest, request: Request, db: Session = Depends(get_db)):
    metadata = body.metadata.model_dump() if body.metadata else None
    success, error = auth_service.signup(db, body.username, body.email, body.password, metadata, request)
    return SignupResponse(success=success, error_message=error)


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    token, error = auth_service.login(db, body.username, body.password, request)
    return LoginResponse(session_token=token, error_message=error)


@router.post("/logout", response_model=SuccessResponse)
def logout(body: LogoutRequest, db: Session = Depends(get_db)):
    auth_service.logout(db, body.session_token)
    return SuccessResponse(success=True)


@router.post("/refresh", response_model=RefreshResponse)
def refresh(body: RefreshRequest, request: Request, db: Session = Depends(get_db)):
    success, token, exp = auth_service.refresh(db, body.session_token, request)
    return RefreshResponse(success=success, session_token=token, expiration_time=exp)


@router.post("/forgot-password", response_model=SuccessResponse)
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    auth_service.forgot_password(db, body.email)
    return SuccessResponse(success=True)


@router.post("/reset-password", response_model=SuccessResponse)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    success, error = auth_service.reset_password(db, body.reset_token, body.new_password)
    return SuccessResponse(success=success, error_message=error)


@router.post("/verify-email", response_model=SuccessResponse)
def verify_email(body: VerifyEmailRequest, db: Session = Depends(get_db)):
    success, error = auth_service.verify_email(db, body.signup_token)
    return SuccessResponse(success=success, error_message=error)


@router.post("/sso/apple", response_model=SSOResponse)
def sso_apple(body: SSORequest, request: Request, db: Session = Depends(get_db)):
    meta = body.client_metadata.model_dump() if body.client_metadata else None
    token, success, error = auth_service.sso_login(db, "apple", body.oauth_token, request, meta)
    return SSOResponse(session_token=token, success=success, error_message=error)


@router.post("/sso/google", response_model=SSOResponse)
def sso_google(body: SSORequest, request: Request, db: Session = Depends(get_db)):
    meta = body.client_metadata.model_dump() if body.client_metadata else None
    token, success, error = auth_service.sso_login(db, "google", body.oauth_token, request, meta)
    return SSOResponse(session_token=token, success=success, error_message=error)


@router.get("/sessions", response_model=list[ActiveSession])
def list_sessions(
    db: Session = Depends(get_db),
    athlete: Athlete = Depends(get_current_athlete),
    session: AuthSession = Depends(get_current_session),
):
    sessions = auth_service.list_sessions(db, athlete.id, session.id)
    return [
        ActiveSession(
            session_id=s.id,
            client_metadata=ClientMetadata(**s.client_metadata),
            ip_address=s.ip_address,
            location=s.location,
            last_active_at=s.last_active_at.isoformat(),
            created_at=s.created_at.isoformat(),
            is_current=s.id == session.id,
        )
        for s in sessions
    ]


@router.delete("/sessions", response_model=SuccessResponse)
def revoke_all_sessions(
    terminate_current: bool = False,
    db: Session = Depends(get_db),
    athlete: Athlete = Depends(get_current_athlete),
    session: AuthSession = Depends(get_current_session),
):
    success, error = auth_service.revoke_all_sessions(db, athlete.id, session.id, terminate_current)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
def revoke_session(
    session_id: str,
    db: Session = Depends(get_db),
    athlete: Athlete = Depends(get_current_athlete),
):
    success, error = auth_service.revoke_session(db, athlete.id, session_id)
    return SuccessResponse(success=success, error_message=error)
