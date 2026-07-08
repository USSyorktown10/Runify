from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.activity import Activity
from app.models.athlete import Athlete
from app.models.social import AthletePost, Club, Comment, Notification
from app.schemas.notification import Notification as NotificationSchema
from app.schemas.notification import NotificationPayload
from app.schemas.notification import NotificationTarget
from app.services.athlete_service import to_summary


def _truncate(text: str, max_len: int = 80) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}…"


def _sender_display_name(athlete: Athlete | None) -> str:
    if not athlete:
        return "Someone"
    name = f"{athlete.first_name} {athlete.last_name}".strip()
    return name or "Someone"


def _format_duration(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_distance(meters: float) -> str:
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{meters:.0f} m"


def _activity_subtitle(activity: Activity) -> str:
    type_label = activity.activity_type.replace("_", " ").title()
    return f"{_format_distance(activity.distance)} · {_format_duration(activity.moving_time)} · {type_label}"


def _activity_target(activity: Activity) -> NotificationTarget:
    return NotificationTarget(
        kind="activity",
        id=activity.id,
        title=activity.name,
        subtitle=_activity_subtitle(activity),
        activity_type=activity.activity_type,
    )


def _club_target(club: Club) -> NotificationTarget:
    privacy = "Private" if club.is_private else "Public"
    members = "member" if club.member_count == 1 else "members"
    subtitle = f"{club.member_count} {members} · {privacy}"
    detail = _truncate(club.description, 120) if club.description.strip() else None
    return NotificationTarget(
        kind="club",
        id=club.id,
        title=club.name,
        subtitle=subtitle,
        detail=detail,
        image_url=club.profile_picture_url or None,
    )


def _post_target(post: AthletePost) -> NotificationTarget:
    detail = _truncate(post.text, 160) if post.text.strip() else None
    return NotificationTarget(
        kind="post",
        id=post.id,
        title="Your post",
        detail=detail,
    )


def _athlete_target(athlete: Athlete) -> NotificationTarget:
    location = ", ".join(p for p in (athlete.city, athlete.state, athlete.country) if p)
    return NotificationTarget(
        kind="athlete",
        id=athlete.id,
        title=_sender_display_name(athlete),
        subtitle=location or None,
        image_url=athlete.profile_picture_url or None,
    )


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

    def _build_context(
        self,
        n: Notification,
        senders: dict[str, Athlete],
        activities: dict[str, Activity],
        clubs: dict[str, Club],
        posts: dict[str, AthletePost],
        comments: dict[str, Comment],
    ) -> tuple[str, str, str | None, NotificationTarget | None]:
        payload = n.payload or {}
        sender = senders.get(n.sender_id) if n.sender_id else None
        sender_name = _sender_display_name(sender)
        ntype = n.type

        activity_id = payload.get("activity_id")
        club_id = payload.get("club_id")
        post_id = payload.get("post_id")
        comment_id = payload.get("comment_id")
        follower_id = payload.get("follower_id") or n.sender_id

        if ntype in {"follow_request", "follow"}:
            link = "/settings/follow-requests" if ntype == "follow_request" else f"/athletes/{follower_id}"
            if ntype == "follow_request":
                message = f"{sender_name} requested to follow you"
            else:
                message = f"{sender_name} started following you"
            target = _athlete_target(sender) if sender else None
            return message, link, None, target

        if ntype in {"activity_like", "like"}:
            activity = activities.get(activity_id) if activity_id else None
            message = f"{sender_name} liked your activity"
            link = f"/activities/{activity_id}" if activity_id else "/activities"
            target = _activity_target(activity) if activity else None
            return message, link, None, target

        if ntype in {"activity_comment", "comment"}:
            activity = activities.get(activity_id) if activity_id else None
            comment = comments.get(comment_id) if comment_id else None
            message = f"{sender_name} commented on your activity"
            excerpt = _truncate(comment.text, 200) if comment else None
            link = f"/activities/{activity_id}" if activity_id else "/activities"
            target = _activity_target(activity) if activity else None
            return message, link, excerpt, target

        if ntype == "post_like":
            post = posts.get(post_id) if post_id else None
            message = f"{sender_name} liked your post"
            owner_id = post.athlete_id if post else n.athlete_id
            link = f"/athletes/{owner_id}/feed"
            target = _post_target(post) if post else None
            return message, link, None, target

        if ntype == "club_invite":
            club = clubs.get(club_id) if club_id else None
            message = f"{sender_name} invited you to join a club"
            link = f"/clubs/{club_id}" if club_id else "/clubs"
            target = _club_target(club) if club else None
            return message, link, None, target

        if ntype == "club_join_request":
            club = clubs.get(club_id) if club_id else None
            message = f"{sender_name} requested to join your club"
            link = f"/clubs/{club_id}/join-requests" if club_id else "/clubs"
            target = _club_target(club) if club else None
            return message, link, None, target

        fallback = ntype.replace("_", " ")
        return f"{sender_name}: {fallback}", "/notifications", None, None

    def to_schema(self, db: Session, n: Notification) -> NotificationSchema:
        return self.to_schemas(db, [n])[0]

    def to_schemas(self, db: Session, notifications: list[Notification]) -> list[NotificationSchema]:
        if not notifications:
            return []

        sender_ids: set[str] = set()
        activity_ids: set[str] = set()
        club_ids: set[str] = set()
        post_ids: set[str] = set()
        comment_ids: set[str] = set()

        for n in notifications:
            if n.sender_id:
                sender_ids.add(n.sender_id)
            payload = n.payload or {}
            if payload.get("activity_id"):
                activity_ids.add(payload["activity_id"])
            if payload.get("club_id"):
                club_ids.add(payload["club_id"])
            if payload.get("post_id"):
                post_ids.add(payload["post_id"])
            if payload.get("comment_id"):
                comment_ids.add(payload["comment_id"])
            if payload.get("follower_id"):
                sender_ids.add(payload["follower_id"])

        senders = {
            a.id: a
            for a in db.scalars(select(Athlete).where(Athlete.id.in_(sender_ids))).all()
        } if sender_ids else {}
        activities = {
            a.id: a
            for a in db.scalars(select(Activity).where(Activity.id.in_(activity_ids))).all()
        } if activity_ids else {}
        clubs = {
            c.id: c for c in db.scalars(select(Club).where(Club.id.in_(club_ids))).all()
        } if club_ids else {}
        posts = {
            p.id: p for p in db.scalars(select(AthletePost).where(AthletePost.id.in_(post_ids))).all()
        } if post_ids else {}
        comments = {
            c.id: c for c in db.scalars(select(Comment).where(Comment.id.in_(comment_ids))).all()
        } if comment_ids else {}

        result: list[NotificationSchema] = []
        for n in notifications:
            sender = senders.get(n.sender_id) if n.sender_id else None
            message, link_path, excerpt, target = self._build_context(
                n, senders, activities, clubs, posts, comments
            )
            result.append(
                NotificationSchema(
                    id=n.id,
                    type=n.type,
                    is_read=n.is_read,
                    created_at=n.created_at.isoformat(),
                    sender_id=n.sender_id,
                    sender=to_summary(sender) if sender else None,
                    message=message,
                    link_path=link_path,
                    excerpt=excerpt,
                    target=target,
                    payload=NotificationPayload(**(n.payload or {})),
                )
            )
        return result

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
