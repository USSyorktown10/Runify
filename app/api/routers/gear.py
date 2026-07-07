from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_athlete
from app.db.session import get_db
from app.models.athlete import Athlete
from app.schemas.common import SuccessResponse
from app.schemas.gear import CreateGearRequest, Gear, UpdateGearRequest
from app.services.gear_service import gear_service

router = APIRouter(prefix="/gear", tags=["gear"])


@router.get("", response_model=list[Gear])
def list_gear(athlete: Athlete = Depends(get_current_athlete), db: Session = Depends(get_db)):
    return gear_service.list_gear(db, athlete.id)


@router.post("")
def create_gear(
    body: CreateGearRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    gear = gear_service.create(db, athlete.id, body.model_dump())
    return {"gear": gear, "success": True}


@router.get("/{gear_id}", response_model=Gear)
def get_gear(
    gear_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    return gear_service.get(db, gear_id, athlete.id)


@router.patch("/{gear_id}", response_model=SuccessResponse)
def update_gear(
    gear_id: str,
    body: UpdateGearRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = gear_service.update(db, gear_id, athlete.id, body.model_dump(exclude_unset=True))
    return SuccessResponse(success=success, error_message=error)


@router.delete("/{gear_id}", response_model=SuccessResponse)
def delete_gear(
    gear_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = gear_service.delete(db, gear_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)
