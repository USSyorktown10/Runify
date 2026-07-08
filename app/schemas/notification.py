from pydantic import BaseModel

from app.schemas.athlete import SummaryAthlete
from app.schemas.common import PaginatedResponseMetadata


class NotificationPayload(BaseModel):
    follower_id: str | None = None
    activity_id: str | None = None
    comment_id: str | None = None
    club_id: str | None = None
    post_id: str | None = None


class NotificationTarget(BaseModel):
    kind: str
    id: str
    title: str
    subtitle: str | None = None
    detail: str | None = None
    image_url: str | None = None
    activity_type: str | None = None


class Notification(BaseModel):
    id: str
    type: str
    is_read: bool
    created_at: str
    sender_id: str | None
    sender: SummaryAthlete | None = None
    message: str
    link_path: str
    excerpt: str | None = None
    target: NotificationTarget | None = None
    payload: NotificationPayload


class PaginatedNotificationsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[Notification]


class UnreadCountResponse(BaseModel):
    unread_count: int


class MarkReadRequest(BaseModel):
    notification_ids: list[str]
