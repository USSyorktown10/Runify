from pydantic import BaseModel

from app.schemas.common import PaginatedResponseMetadata


class NotificationPayload(BaseModel):
    follower_id: str | None = None
    activity_id: str | None = None
    comment_id: str | None = None
    club_id: str | None = None


class Notification(BaseModel):
    id: str
    type: str
    is_read: bool
    created_at: str
    sender_id: str | None
    payload: NotificationPayload


class PaginatedNotificationsResponse(BaseModel):
    pagination: PaginatedResponseMetadata
    items: list[Notification]


class UnreadCountResponse(BaseModel):
    unread_count: int


class MarkReadRequest(BaseModel):
    notification_ids: list[str]
