from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.models.activity import Activity, Gear
from app.schemas.gear import Gear as GearSchema


class GearService:
    def list_gear(self, db: Session, athlete_id: str) -> list[GearSchema]:
        gears = db.scalars(select(Gear).where(Gear.athlete_id == athlete_id)).all()
        return [self._to_schema(db, g) for g in gears]

    def _total_mileage(self, db: Session, gear_id: str) -> float:
        return db.scalar(
            select(func.coalesce(func.sum(Activity.distance), 0.0)).where(Activity.gear_id == gear_id)
        ) or 0.0

    def _to_schema(self, db: Session, gear: Gear) -> GearSchema:
        return GearSchema(
            id=gear.id,
            name=gear.name,
            brand_name=gear.brand_name,
            model_name=gear.model_name,
            gear_type=gear.gear_type,
            is_primary=gear.is_primary,
            max_mileage=gear.max_mileage,
            total_mileage=self._total_mileage(db, gear.id),
            is_retired=gear.is_retired,
            initial_date=gear.initial_date.isoformat() if gear.initial_date else None,
            created_date=gear.created_date.isoformat(),
        )

    def create(self, db: Session, athlete_id: str, data: dict) -> GearSchema:
        from datetime import datetime

        gear = Gear(
            athlete_id=athlete_id,
            name=data["name"],
            brand_name=data.get("brand_name", ""),
            model_name=data.get("model_name", ""),
            max_mileage=data.get("max_mileage", 0.0),
        )
        if data.get("initial_date"):
            gear.initial_date = datetime.fromisoformat(data["initial_date"])
        db.add(gear)
        db.commit()
        db.refresh(gear)
        return self._to_schema(db, gear)

    def get(self, db: Session, gear_id: str, athlete_id: str) -> GearSchema:
        gear = db.get(Gear, gear_id)
        if not gear or gear.athlete_id != athlete_id:
            raise NotFoundError("Gear not found")
        return self._to_schema(db, gear)

    def update(self, db: Session, gear_id: str, athlete_id: str, data: dict) -> tuple[bool, str | None]:
        gear = db.get(Gear, gear_id)
        if not gear or gear.athlete_id != athlete_id:
            return False, "Gear not found"
        for field in ("name", "is_primary", "is_retired", "max_mileage"):
            if field in data and data[field] is not None:
                setattr(gear, field, data[field])
        if data.get("is_primary"):
            others = db.scalars(select(Gear).where(Gear.athlete_id == athlete_id, Gear.id != gear_id)).all()
            for o in others:
                o.is_primary = False
        db.commit()
        return True, None

    def delete(self, db: Session, gear_id: str, athlete_id: str) -> tuple[bool, str | None]:
        gear = db.get(Gear, gear_id)
        if not gear or gear.athlete_id != athlete_id:
            return False, "Gear not found"
        db.delete(gear)
        db.commit()
        return True, None


gear_service = GearService()
