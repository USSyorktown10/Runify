from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    skip: int
    limit: int

class Message(BaseModel):
    detail: str