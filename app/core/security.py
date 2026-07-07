import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, Header, Request
from sqlalchemy import select
from sqlalchemy.orm import Session
from user_agents import parse as parse_user_agent

from app.core.config import get_settings
from app.core.errors import UnauthorizedError
from app.db.session import get_db
from app.models.athlete import Athlete
from app.models.auth import AuthSession


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def parse_client_metadata(request: Request, override: dict | None = None) -> dict:
    if override:
        return override
    ua_string = request.headers.get("user-agent", "")
    ua = parse_user_agent(ua_string)
    return {
        "user_agent": ua_string,
        "browser_name": ua.browser.family or "Unknown",
        "browser_version": ua.browser.version_string or "",
        "os_name": ua.os.family or "Unknown",
    }


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def create_session(
    db: Session,
    athlete: Athlete,
    request: Request,
    client_metadata: dict | None = None,
) -> tuple[str, AuthSession]:
    settings = get_settings()
    raw_token = generate_token()
    session = AuthSession(
        athlete_id=athlete.id,
        token_hash=hash_token(raw_token),
        client_metadata=parse_client_metadata(request, client_metadata),
        ip_address=get_client_ip(request),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.session_expire_days),
    )
    db.add(session)
    db.flush()
    return raw_token, session


def resolve_session(db: Session, token: str) -> AuthSession | None:
    token_hash = hash_token(token)
    session = db.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == token_hash,
            AuthSession.revoked_at.is_(None),
        )
    )
    if not session:
        return None
    if session.expires_at.tzinfo is None:
        expires = session.expires_at.replace(tzinfo=timezone.utc)
    else:
        expires = session.expires_at
    if expires < datetime.now(timezone.utc):
        return None
    return session


PUBLIC_PREFIXES = (
    "/authentication/login",
    "/authentication/signup",
    "/authentication/logout",
    "/authentication/refresh",
    "/authentication/forgot-password",
    "/authentication/reset-password",
    "/authentication/verify-email",
    "/authentication/sso/",
    "/webhooks/",
    "/integrations/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/health",
)


def is_public_path(path: str) -> bool:
    if path == "/authentication/sessions" or path.startswith("/authentication/sessions/"):
        return False
    for prefix in PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def get_optional_athlete(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> Athlete | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    session = resolve_session(db, token)
    if not session:
        return None
    return db.get(Athlete, session.athlete_id)


def get_current_athlete(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> Athlete:
    if not authorization or not authorization.startswith("Bearer "):
        raise UnauthorizedError("Missing or invalid authorization header")
    token = authorization[7:]
    session = resolve_session(db, token)
    if not session:
        raise UnauthorizedError("Invalid or expired session token")
    athlete = db.get(Athlete, session.athlete_id)
    if not athlete:
        raise UnauthorizedError("Athlete not found")
    return athlete


def get_current_session(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> AuthSession:
    if not authorization or not authorization.startswith("Bearer "):
        raise UnauthorizedError("Missing or invalid authorization header")
    token = authorization[7:]
    session = resolve_session(db, token)
    if not session:
        raise UnauthorizedError("Invalid or expired session token")
    return session
