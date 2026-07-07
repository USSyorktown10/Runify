from datetime import date, datetime, timedelta, timezone

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import (
    create_session,
    generate_token,
    hash_password,
    hash_token,
    resolve_session,
    verify_password,
)
from app.models.athlete import Athlete, AthletePreferences, AthleteStats
from app.models.auth import AuthSession, AuthToken
from app.services.email_service import email_service


class AuthService:
    def signup(
        self,
        db: Session,
        username: str,
        email: str,
        password: str,
        metadata: dict | None,
        request: Request,
    ) -> tuple[bool, str | None]:
        if db.scalar(select(Athlete).where(Athlete.username == username)):
            return False, "Username already taken"
        if db.scalar(select(Athlete).where(Athlete.email == email)):
            return False, "Email already registered"

        athlete = Athlete(
            username=username,
            email=email,
            password_hash=hash_password(password),
            first_name=username,
        )
        if metadata:
            athlete.gender = metadata.get("gender")
            if metadata.get("birthdate"):
                athlete.birthdate = date.fromisoformat(metadata["birthdate"])
            athlete.weight_kg = metadata.get("weight_kg")
            athlete.height_cm = metadata.get("height_cm")

        db.add(athlete)
        db.flush()

        db.add(AthletePreferences(athlete_id=athlete.id))
        db.add(AthleteStats(athlete_id=athlete.id))

        token = generate_token()
        db.add(
            AuthToken(
                athlete_id=athlete.id,
                token_hash=hash_token(token),
                token_type="signup",
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            )
        )
        email_service.send(
            db,
            email,
            "Verify your Runify account",
            f"Your verification token: {token}",
        )
        db.commit()
        return True, None

    def login(self, db: Session, username: str, password: str, request: Request) -> tuple[str | None, str | None]:
        athlete = db.scalar(
            select(Athlete).where((Athlete.username == username) | (Athlete.email == username))
        )
        if not athlete or not verify_password(password, athlete.password_hash):
            return None, "Invalid credentials"
        raw_token, _ = create_session(db, athlete, request)
        db.commit()
        return raw_token, None

    def logout(self, db: Session, session_token: str) -> bool:
        session = resolve_session(db, session_token)
        if session:
            session.revoked_at = datetime.now(timezone.utc)
            db.commit()
        return True

    def refresh(self, db: Session, session_token: str, request: Request) -> tuple[bool, str | None, str | None]:
        session = resolve_session(db, session_token)
        if not session:
            return False, None, None
        athlete = db.get(Athlete, session.athlete_id)
        if not athlete:
            return False, None, None
        session.revoked_at = datetime.now(timezone.utc)
        raw_token, new_session = create_session(db, athlete, request)
        db.commit()
        return True, raw_token, new_session.expires_at.isoformat()

    def forgot_password(self, db: Session, email: str) -> bool:
        athlete = db.scalar(select(Athlete).where(Athlete.email == email))
        if athlete:
            token = generate_token()
            db.add(
                AuthToken(
                    athlete_id=athlete.id,
                    token_hash=hash_token(token),
                    token_type="password_reset",
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                )
            )
            email_service.send(db, email, "Reset your Runify password", f"Reset token: {token}")
            db.commit()
        return True

    def reset_password(self, db: Session, reset_token: str, new_password: str) -> tuple[bool, str | None]:
        token_hash = hash_token(reset_token)
        auth_token = db.scalar(
            select(AuthToken).where(
                AuthToken.token_hash == token_hash,
                AuthToken.token_type == "password_reset",
                AuthToken.used_at.is_(None),
            )
        )
        if not auth_token or auth_token.expires_at < datetime.now(timezone.utc):
            return False, "Invalid or expired reset token"
        athlete = db.get(Athlete, auth_token.athlete_id)
        if not athlete:
            return False, "Athlete not found"
        athlete.password_hash = hash_password(new_password)
        auth_token.used_at = datetime.now(timezone.utc)
        db.commit()
        return True, None

    def verify_email(self, db: Session, signup_token: str) -> tuple[bool, str | None]:
        token_hash = hash_token(signup_token)
        auth_token = db.scalar(
            select(AuthToken).where(
                AuthToken.token_hash == token_hash,
                AuthToken.token_type == "signup",
                AuthToken.used_at.is_(None),
            )
        )
        if not auth_token or auth_token.expires_at < datetime.now(timezone.utc):
            return False, "Invalid or expired verification token"
        athlete = db.get(Athlete, auth_token.athlete_id)
        if not athlete:
            return False, "Athlete not found"
        athlete.email_verified = True
        auth_token.used_at = datetime.now(timezone.utc)
        db.commit()
        return True, None

    def sso_login(
        self,
        db: Session,
        provider: str,
        oauth_token: str,
        request: Request,
        client_metadata: dict | None,
    ) -> tuple[str | None, bool, str | None]:
        email = f"{provider}_{hash_token(oauth_token)[:12]}@sso.runify.local"
        username = f"{provider}_{hash_token(oauth_token)[:8]}"
        athlete = db.scalar(select(Athlete).where(Athlete.email == email))
        if not athlete:
            athlete = Athlete(
                username=username,
                email=email,
                password_hash=hash_password(generate_token()),
                email_verified=True,
                first_name=provider.title(),
            )
            db.add(athlete)
            db.flush()
            db.add(AthletePreferences(athlete_id=athlete.id))
            db.add(AthleteStats(athlete_id=athlete.id))
        raw_token, _ = create_session(db, athlete, request, client_metadata)
        db.commit()
        return raw_token, True, None

    def list_sessions(self, db: Session, athlete_id: str, current_session_id: str) -> list[AuthSession]:
        return list(
            db.scalars(
                select(AuthSession).where(
                    AuthSession.athlete_id == athlete_id,
                    AuthSession.revoked_at.is_(None),
                ).order_by(AuthSession.last_active_at.desc())
            ).all()
        )

    def revoke_session(self, db: Session, athlete_id: str, session_id: str) -> tuple[bool, str | None]:
        session = db.get(AuthSession, session_id)
        if not session or session.athlete_id != athlete_id:
            return False, "Session not found"
        session.revoked_at = datetime.now(timezone.utc)
        db.commit()
        return True, None

    def revoke_all_sessions(
        self, db: Session, athlete_id: str, current_session_id: str, terminate_current: bool
    ) -> tuple[bool, str | None]:
        sessions = self.list_sessions(db, athlete_id, current_session_id)
        now = datetime.now(timezone.utc)
        for s in sessions:
            if not terminate_current and s.id == current_session_id:
                continue
            s.revoked_at = now
        db.commit()
        return True, None


auth_service = AuthService()
