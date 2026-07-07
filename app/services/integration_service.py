import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import generate_token
from app.models.social import Integration, Report

PROVIDERS = ["garmin", "wahoo", "apple_health"]


class IntegrationService:
    def list_status(self, db: Session, athlete_id: str) -> list[dict]:
        integrations = {
            i.provider: i
            for i in db.scalars(select(Integration).where(Integration.athlete_id == athlete_id)).all()
        }
        result = []
        for p in PROVIDERS:
            i = integrations.get(p)
            result.append(
                {
                    "provider": p,
                    "is_connected": i is not None and i.connected_at is not None,
                    "connected_at": i.connected_at.isoformat() if i and i.connected_at else None,
                }
            )
        return result

    def connect(self, db: Session, athlete_id: str, provider: str) -> str:
        state = secrets.token_urlsafe(16)
        existing = db.scalar(
            select(Integration).where(Integration.athlete_id == athlete_id, Integration.provider == provider)
        )
        if existing:
            existing.oauth_state = state
        else:
            db.add(Integration(athlete_id=athlete_id, provider=provider, oauth_state=state))
        db.commit()
        return f"https://oauth.runify.local/{provider}/authorize?state={state}&athlete={athlete_id}"

    def callback(self, db: Session, provider: str, state: str, external_user_id: str) -> tuple[bool, str | None]:
        integration = db.scalar(
            select(Integration).where(Integration.provider == provider, Integration.oauth_state == state)
        )
        if not integration:
            return False, "Invalid OAuth state"
        integration.connected_at = datetime.now(timezone.utc)
        integration.external_user_id = external_user_id
        integration.access_token = generate_token()
        integration.oauth_state = None
        db.commit()
        return True, None

    def disconnect(self, db: Session, athlete_id: str, provider: str) -> bool:
        integration = db.scalar(
            select(Integration).where(Integration.athlete_id == athlete_id, Integration.provider == provider)
        )
        if integration:
            db.delete(integration)
            db.commit()
        return True

    def resolve_athlete_by_external(self, db: Session, provider: str, external_user_id: str) -> str | None:
        integration = db.scalar(
            select(Integration).where(
                Integration.provider == provider,
                Integration.external_user_id == external_user_id,
            )
        )
        return integration.athlete_id if integration else None

    def create_report(
        self, db: Session, reporter_id: str, target_type: str, target_id: str, reason: str, details: str
    ) -> tuple[bool, str | None]:
        db.add(
            Report(
                reporter_id=reporter_id,
                target_type=target_type,
                target_id=target_id,
                reason=reason,
                details=details,
            )
        )
        db.commit()
        return True, None


integration_service = IntegrationService()
