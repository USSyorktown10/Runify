from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.schemas.common import SuccessResponse
from app.schemas.integration import ConnectResponse, IntegrationStatus
from app.services.integration_service import integration_service

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("", response_model=list[IntegrationStatus])
def list_integrations(athlete: Athlete = Depends(get_current_athlete), db: Session = Depends(get_db)):
    return [IntegrationStatus(**s) for s in integration_service.list_status(db, athlete.id)]


@router.get("/{provider}/connect", response_model=ConnectResponse)
def connect_provider(
    provider: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    url = integration_service.connect(db, athlete.id, provider)
    return ConnectResponse(redirect_url=url)


@router.get("/{provider}/callback", response_model=SuccessResponse)
def oauth_callback(
    provider: str,
    state: str = Query(...),
    user_id: str = Query("external"),
    db: Session = Depends(get_db),
):
    success, error = integration_service.callback(db, provider, state, user_id)
    return SuccessResponse(success=success, error_message=error)


@router.delete("/{provider}", response_model=SuccessResponse)
def disconnect_provider(
    provider: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    integration_service.disconnect(db, athlete.id, provider)
    return SuccessResponse(success=True)
