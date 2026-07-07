from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.social import Notification
from app.schemas.notification import Notification as NotificationSchema
from app.schemas.notification import NotificationPayload


class NotificationService:
    def create(
        self,
        db: Session,
        athlete_id: str,
        ntype: str,
        sender_id: str | None,
        payload: dict,
    ) -> Notification:
        notif = Notification(
            athlete_id=athlete_id,
            type=ntype,
            sender_id=sender_id,
            payload=payload,
        )
        db.add(notif)
        db.flush()
        return notif

    def to_schema(self, n: Notification) -> NotificationSchema:
        return NotificationSchema(
            id=n.id,
            type=n.type,
            is_read=n.is_read,
            created_at=n.created_at.isoformat(),
            sender_id=n.sender_id,
            payload=NotificationPayload(**n.payload),
        )

    def list_notifications(self, db: Session, athlete_id: str, page: int, per_page: int):
        return (
            select(Notification)
            .where(Notification.athlete_id == athlete_id)
            .order_by(Notification.created_at.desc())
        )

    def unread_count(self, db: Session, athlete_id: str) -> int:
        return db.scalar(
            select(func.count()).select_from(Notification).where(
                Notification.athlete_id == athlete_id, Notification.is_read.is_(False)
            )
        ) or 0

    def mark_read(self, db: Session, athlete_id: str, notification_ids: list[str]) -> tuple[bool, str | None]:
        notifs = db.scalars(
            select(Notification).where(
                Notification.athlete_id == athlete_id, Notification.id.in_(notification_ids)
            )
        ).all()
        for n in notifs:
            n.is_read = True
        db.commit()
        return True, None

    def mark_all_read(self, db: Session, athlete_id: str) -> tuple[bool, str | None]:
        notifs = db.scalars(
            select(Notification).where(Notification.athlete_id == athlete_id, Notification.is_read.is_(False))
        ).all()
        for n in notifs:
            n.is_read = True
        db.commit()
        return True, None

    def delete(self, db: Session, athlete_id: str, notification_id: str) -> tuple[bool, str | None]:
        notif = db.get(Notification, notification_id)
        if not notif or notif.athlete_id != athlete_id:
            return False, "Notification not found"
        db.delete(notif)
        db.commit()
        return True, None


notification_service = NotificationService()
