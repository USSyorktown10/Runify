import base64
from datetime import datetime
from typing import TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.schemas.common import PaginatedResponseMetadata

T = TypeVar("T")


def paginate_offset(
    db: Session,
    stmt: Select,
    page: int = 1,
    per_page: int = 20,
    unique: bool = False,
) -> tuple[list, PaginatedResponseMetadata]:
    page = max(1, page)
    per_page = min(max(1, per_page), 100)
    total_items = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    total_pages = max(1, (total_items + per_page - 1) // per_page) if total_items else 0
    scalars = db.scalars(stmt.offset((page - 1) * per_page).limit(per_page))
    if unique:
        scalars = scalars.unique()
    items = list(scalars.all())
    return items, PaginatedResponseMetadata(
        page=page,
        per_page=per_page,
        total_items=total_items,
        total_pages=total_pages,
    )


def encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode()


def decode_cursor(cursor: str | None) -> str | None:
    if not cursor:
        return None
    try:
        return base64.urlsafe_b64decode(cursor.encode()).decode()
    except Exception:
        return None


def encode_datetime_cursor(dt: datetime) -> str:
    return encode_cursor(dt.isoformat())


def decode_datetime_cursor(cursor: str | None) -> datetime | None:
    raw = decode_cursor(cursor)
    if not raw:
        return None
    return datetime.fromisoformat(raw)
