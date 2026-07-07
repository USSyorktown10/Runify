from pydantic import BaseModel


class Gear(BaseModel):
    id: str
    name: str
    brand_name: str
    model_name: str
    gear_type: str
    is_primary: bool
    max_mileage: float
    total_mileage: float
    is_retired: bool
    initial_date: str | None
    created_date: str


class CreateGearRequest(BaseModel):
    name: str
    brand_name: str = ""
    model_name: str = ""
    initial_date: str | None = None
    max_mileage: float = 0.0


class UpdateGearRequest(BaseModel):
    name: str | None = None
    is_primary: bool | None = None
    is_retired: bool | None = None
    max_mileage: float | None = None
