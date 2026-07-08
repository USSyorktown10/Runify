from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.errors import ForbiddenError, NotFoundError
from app.core.pagination import paginate_offset
from app.models.activity import Activity, ActivityMetric
from app.models.athlete import Athlete
from app.models.social import Club, ClubInvite, ClubJoinRequest, ClubMember, ClubPost, Follow, Like
from app.schemas.activity import SummaryActivity
from app.schemas.club import (
    ClubLeaderboardEntry,
    ClubLeaderboardSummary,
    DetailedClub,
    SummaryClub,
)
from app.schemas.common import PaginatedResponseMetadata
from app.schemas.social import Post
from app.services.activity_service import activity_service
from app.services.athlete_service import to_summary
from app.services.notification_service import notification_service


class ClubService:
    def to_summary(self, club: Club) -> SummaryClub:
        return SummaryClub(
            id=club.id,
            name=club.name,
            profile_picture_url=club.profile_picture_url,
            member_count=club.member_count,
            is_private=club.is_private,
        )

    def _get_member(self, db: Session, club_id: str, athlete_id: str) -> ClubMember | None:
        return db.scalar(
            select(ClubMember).where(ClubMember.club_id == club_id, ClubMember.athlete_id == athlete_id)
        )

    def _is_member(self, db: Session, club_id: str, athlete_id: str) -> bool:
        return self._get_member(db, club_id, athlete_id) is not None

    def _can_read_club_content(self, db: Session, club: Club, viewer_id: str | None) -> bool:
        if not club.is_private:
            return True
        if not viewer_id:
            return False
        return self._is_member(db, club.id, viewer_id)

    def to_detailed(self, db: Session, club: Club, viewer_id: str | None = None) -> DetailedClub:
        admins = db.scalars(
            select(ClubMember.athlete_id).where(
                ClubMember.club_id == club.id, ClubMember.role.in_(["admin", "owner"])
            )
        ).all()
        is_member = False
        viewer_role = None
        has_pending_join_request = False
        has_pending_invite = False
        if viewer_id:
            member = self._get_member(db, club.id, viewer_id)
            if member:
                is_member = True
                viewer_role = member.role
            else:
                has_pending_join_request = (
                    db.scalar(
                        select(ClubJoinRequest).where(
                            ClubJoinRequest.club_id == club.id,
                            ClubJoinRequest.athlete_id == viewer_id,
                            ClubJoinRequest.status == "pending",
                        )
                    )
                    is not None
                )
                has_pending_invite = (
                    db.scalar(
                        select(ClubInvite).where(
                            ClubInvite.club_id == club.id,
                            ClubInvite.athlete_id == viewer_id,
                            ClubInvite.status == "pending",
                        )
                    )
                    is not None
                )
        return DetailedClub(
            id=club.id,
            name=club.name,
            description=club.description,
            profile_picture_url=club.profile_picture_url,
            cover_photo_url=club.cover_photo_url,
            member_count=club.member_count,
            is_private=club.is_private,
            creator_id=club.creator_id,
            created_at=club.created_at.isoformat(),
            admins=list(admins),
            tags=club.tags or [],
            is_member=is_member,
            viewer_role=viewer_role,
            has_pending_join_request=has_pending_join_request,
            has_pending_invite=has_pending_invite,
        )

    def create(self, db: Session, creator_id: str, data: dict) -> DetailedClub:
        club = Club(
            name=data["name"],
            description=data.get("description", ""),
            is_private=data.get("is_private", False),
            creator_id=creator_id,
            tags=data.get("tags", []),
        )
        db.add(club)
        db.flush()
        db.add(ClubMember(club_id=club.id, athlete_id=creator_id, role="owner"))
        db.commit()
        db.refresh(club)
        return self.to_detailed(db, club, creator_id)

    def _is_admin(self, db: Session, club_id: str, athlete_id: str) -> bool:
        member = self._get_member(db, club_id, athlete_id)
        return member is not None and member.role in ("owner", "admin")

    def join(self, db: Session, club_id: str, athlete_id: str) -> tuple[bool, str | None]:
        club = db.get(Club, club_id)
        if not club:
            return False, "Club not found"
        existing = self._get_member(db, club_id, athlete_id)
        if existing:
            return True, None
        if club.is_private:
            if not db.scalar(
                select(ClubJoinRequest).where(
                    ClubJoinRequest.club_id == club_id,
                    ClubJoinRequest.athlete_id == athlete_id,
                    ClubJoinRequest.status == "pending",
                )
            ):
                db.add(ClubJoinRequest(club_id=club_id, athlete_id=athlete_id))
                notification_service.create(
                    db, club.creator_id, "club_join_request", athlete_id, {"club_id": club_id}
                )
                db.commit()
            return True, None
        db.add(ClubMember(club_id=club_id, athlete_id=athlete_id, role="member"))
        club.member_count += 1
        db.commit()
        return True, None

    def invite(self, db: Session, club_id: str, inviter_id: str, athlete_id: str) -> tuple[bool, str | None]:
        if not self._is_admin(db, club_id, inviter_id):
            return False, "Admin access required"
        if self._is_member(db, club_id, athlete_id):
            return False, "Athlete is already a member"
        existing = db.scalar(
            select(ClubInvite).where(
                ClubInvite.club_id == club_id,
                ClubInvite.athlete_id == athlete_id,
                ClubInvite.status == "pending",
            )
        )
        if not existing:
            db.add(ClubInvite(club_id=club_id, athlete_id=athlete_id, invited_by=inviter_id))
            notification_service.create(db, athlete_id, "club_invite", inviter_id, {"club_id": club_id})
            db.commit()
        return True, None

    def remove_member(
        self, db: Session, club_id: str, target_id: str, actor_id: str
    ) -> tuple[bool, str | None]:
        club = db.get(Club, club_id)
        if not club:
            return False, "Club not found"
        target_member = self._get_member(db, club_id, target_id)
        if not target_member:
            return False, "Member not found"
        if target_member.role == "owner":
            return False, "Cannot remove club owner"
        if actor_id == target_id:
            if target_member.role == "owner":
                return False, "Owner cannot leave; transfer ownership or delete club"
        elif not self._is_admin(db, club_id, actor_id):
            return False, "Admin access required"
        db.delete(target_member)
        club.member_count = max(0, club.member_count - 1)
        db.commit()
        return True, None

    def list_join_requests(self, db: Session, club_id: str, admin_id: str, page: int, per_page: int):
        if not self._is_admin(db, club_id, admin_id):
            raise ForbiddenError("Admin access required")
        stmt = (
            select(Athlete)
            .join(ClubJoinRequest, ClubJoinRequest.athlete_id == Athlete.id)
            .where(ClubJoinRequest.club_id == club_id, ClubJoinRequest.status == "pending")
            .order_by(ClubJoinRequest.created_at.asc())
        )
        return paginate_offset(db, stmt, page, per_page)

    def accept_join_request(
        self, db: Session, club_id: str, athlete_id: str, admin_id: str
    ) -> tuple[bool, str | None]:
        if not self._is_admin(db, club_id, admin_id):
            return False, "Admin access required"
        request = db.scalar(
            select(ClubJoinRequest).where(
                ClubJoinRequest.club_id == club_id,
                ClubJoinRequest.athlete_id == athlete_id,
                ClubJoinRequest.status == "pending",
            )
        )
        if not request:
            return False, "Join request not found"
        if self._is_member(db, club_id, athlete_id):
            request.status = "accepted"
            db.commit()
            return True, None
        club = db.get(Club, club_id)
        db.add(ClubMember(club_id=club_id, athlete_id=athlete_id, role="member"))
        club.member_count += 1
        request.status = "accepted"
        db.commit()
        return True, None

    def deny_join_request(
        self, db: Session, club_id: str, athlete_id: str, admin_id: str
    ) -> tuple[bool, str | None]:
        if not self._is_admin(db, club_id, admin_id):
            return False, "Admin access required"
        request = db.scalar(
            select(ClubJoinRequest).where(
                ClubJoinRequest.club_id == club_id,
                ClubJoinRequest.athlete_id == athlete_id,
                ClubJoinRequest.status == "pending",
            )
        )
        if not request:
            return False, "Join request not found"
        request.status = "denied"
        db.commit()
        return True, None

    def accept_invite(self, db: Session, club_id: str, athlete_id: str) -> tuple[bool, str | None]:
        invite = db.scalar(
            select(ClubInvite).where(
                ClubInvite.club_id == club_id,
                ClubInvite.athlete_id == athlete_id,
                ClubInvite.status == "pending",
            )
        )
        if not invite:
            return False, "Invite not found"
        if self._is_member(db, club_id, athlete_id):
            invite.status = "accepted"
            db.commit()
            return True, None
        club = db.get(Club, club_id)
        if not club:
            return False, "Club not found"
        db.add(ClubMember(club_id=club_id, athlete_id=athlete_id, role="member"))
        club.member_count += 1
        invite.status = "accepted"
        db.commit()
        return True, None

    def deny_invite(self, db: Session, club_id: str, athlete_id: str) -> tuple[bool, str | None]:
        invite = db.scalar(
            select(ClubInvite).where(
                ClubInvite.club_id == club_id,
                ClubInvite.athlete_id == athlete_id,
                ClubInvite.status == "pending",
            )
        )
        if not invite:
            return False, "Invite not found"
        invite.status = "denied"
        db.commit()
        return True, None

    def to_post(self, db: Session, post: ClubPost, viewer_id: str | None = None) -> Post:
        club = post.club if post.club is not None else db.get(Club, post.club_id)
        if not club:
            raise ValueError(f"Club {post.club_id} not found for post {post.id}")
        author = db.get(Athlete, post.author_id)
        is_liked = False
        if viewer_id:
            is_liked = (
                db.scalar(
                    select(Like).where(
                        Like.athlete_id == viewer_id,
                        Like.target_type == "club_post",
                        Like.target_id == post.id,
                    )
                )
                is not None
            )
        return Post(
            id=post.id,
            club_id=post.club_id,
            club=self.to_summary(club),
            author=to_summary(author),
            title=post.title,
            body=post.body,
            created_at=post.created_at.isoformat(),
            like_count=post.like_count,
            comment_count=post.comment_count,
            is_liked=is_liked,
        )

    def create_post(self, db: Session, club_id: str, author_id: str, title: str, body: str) -> Post:
        if not self._is_member(db, club_id, author_id):
            raise ForbiddenError("Must be a club member")
        post = ClubPost(club_id=club_id, author_id=author_id, title=title, body=body)
        db.add(post)
        db.commit()
        db.refresh(post)
        return self.to_post(db, post, author_id)

    def list_posts(
        self, db: Session, club_id: str, viewer_id: str | None, page: int, per_page: int
    ) -> tuple[list[Post], PaginatedResponseMetadata]:
        club = db.get(Club, club_id)
        if not club:
            raise NotFoundError("Club not found")
        if not self._can_read_club_content(db, club, viewer_id):
            raise ForbiddenError("Club membership required")
        stmt = (
            select(ClubPost)
            .options(joinedload(ClubPost.club))
            .where(ClubPost.club_id == club_id)
            .order_by(ClubPost.created_at.desc())
        )
        posts, pagination = paginate_offset(db, stmt, page, per_page, unique=True)
        return [self.to_post(db, p, viewer_id) for p in posts], pagination

    @staticmethod
    def week_window(period: str, now: datetime | None = None) -> tuple[datetime, datetime]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        weekday = now.weekday()
        monday = (now - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "last_week":
            monday = monday - timedelta(days=7)
        end = monday + timedelta(days=7)
        return monday, end

    def _activity_visible(self, db: Session, activity: Activity, viewer_id: str | None) -> bool:
        if activity.visibility == "public":
            return True
        if not viewer_id:
            return False
        if activity.athlete_id == viewer_id:
            return True
        if activity.visibility == "private":
            return False
        return (
            db.scalar(
                select(Follow).where(
                    Follow.follower_id == viewer_id,
                    Follow.following_id == activity.athlete_id,
                    Follow.status == "following",
                )
            )
            is not None
        )

    def _aggregate_leaderboard(
        self, db: Session, club_id: str, period: str
    ) -> list[dict]:
        start, end = self.week_window(period)
        member_ids = list(
            db.scalars(select(ClubMember.athlete_id).where(ClubMember.club_id == club_id)).all()
        )
        if not member_ids:
            return []
        activities = db.scalars(
            select(Activity)
            .options(joinedload(Activity.metrics))
            .where(
                Activity.athlete_id.in_(member_ids),
                Activity.activity_type == "run",
                Activity.start_date >= start,
                Activity.start_date < end,
            )
        ).unique().all()
        elev_by_activity: dict[str, float] = {}
        activity_ids = [a.id for a in activities]
        if activity_ids:
            metrics = db.scalars(
                select(ActivityMetric).where(
                    ActivityMetric.activity_id.in_(activity_ids),
                    ActivityMetric.key == "elevation_gain",
                )
            ).all()
            for m in metrics:
                elev_by_activity[m.activity_id] = elev_by_activity.get(m.activity_id, 0) + m.value

        totals: dict[str, dict] = {}
        for act in activities:
            bucket = totals.setdefault(
                act.athlete_id,
                {
                    "distance": 0.0,
                    "activity_count": 0,
                    "longest_activity_id": None,
                    "longest_distance": 0.0,
                    "total_moving_time": 0,
                    "elevation_gain": 0.0,
                },
            )
            bucket["distance"] += act.distance
            bucket["activity_count"] += 1
            bucket["total_moving_time"] += act.moving_time
            bucket["elevation_gain"] += elev_by_activity.get(act.id, 0.0)
            if act.distance > bucket["longest_distance"]:
                bucket["longest_distance"] = act.distance
                bucket["longest_activity_id"] = act.id
        return [
            {"athlete_id": athlete_id, **data}
            for athlete_id, data in totals.items()
        ]

    @staticmethod
    def _sort_leaderboard(entries: list[dict], metric: str) -> list[dict]:
        if metric == "activity_count":
            return sorted(entries, key=lambda e: (-e["activity_count"], -e["distance"]))
        if metric == "longest_distance":
            return sorted(entries, key=lambda e: (-e["longest_distance"], -e["distance"]))
        if metric == "avg_pace":
            def pace_key(e):
                if e["distance"] <= 0:
                    return (float("inf"),)
                return (e["total_moving_time"] / (e["distance"] / 1000.0),)
            return sorted(entries, key=pace_key)
        if metric == "elevation_gain":
            return sorted(entries, key=lambda e: (-e["elevation_gain"], -e["distance"]))
        return sorted(entries, key=lambda e: (-e["distance"], -e["activity_count"]))

    def leaderboard(
        self,
        db: Session,
        club_id: str,
        viewer_id: str | None,
        period: str,
        metric: str,
        page: int,
        per_page: int,
    ) -> tuple[list[ClubLeaderboardEntry], PaginatedResponseMetadata, ClubLeaderboardSummary | None]:
        club = db.get(Club, club_id)
        if not club:
            raise NotFoundError("Club not found")
        if not self._can_read_club_content(db, club, viewer_id):
            raise ForbiddenError("Club membership required")

        raw = self._aggregate_leaderboard(db, club_id, period)
        sorted_entries = self._sort_leaderboard(raw, metric)
        for i, entry in enumerate(sorted_entries, start=1):
            entry["rank"] = i

        page = max(1, page)
        per_page = min(max(1, per_page), 100)
        total_items = len(sorted_entries)
        total_pages = max(1, (total_items + per_page - 1) // per_page) if total_items else 0
        start_idx = (page - 1) * per_page
        page_slice = sorted_entries[start_idx : start_idx + per_page]

        athlete_ids = [e["athlete_id"] for e in page_slice]
        athletes = {
            a.id: a
            for a in db.scalars(select(Athlete).where(Athlete.id.in_(athlete_ids))).all()
        } if athlete_ids else {}

        items: list[ClubLeaderboardEntry] = []
        for entry in page_slice:
            athlete = athletes.get(entry["athlete_id"])
            if not athlete:
                continue
            avg_pace = None
            if entry["distance"] > 0:
                avg_pace = entry["total_moving_time"] / (entry["distance"] / 1000.0)
            items.append(
                ClubLeaderboardEntry(
                    rank=entry["rank"],
                    athlete_id=entry["athlete_id"],
                    athlete=to_summary(athlete),
                    distance=entry["distance"],
                    activity_count=entry["activity_count"],
                    longest_activity_id=entry["longest_activity_id"],
                    longest_distance=entry["longest_distance"],
                    avg_pace=avg_pace,
                    elevation_gain=entry["elevation_gain"],
                )
            )

        pagination = PaginatedResponseMetadata(
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
        )

        viewer_summary = None
        if viewer_id and self._is_member(db, club_id, viewer_id):
            viewer_entry = next((e for e in sorted_entries if e["athlete_id"] == viewer_id), None)
            if viewer_entry:
                avg_pace = None
                if viewer_entry["distance"] > 0:
                    avg_pace = viewer_entry["total_moving_time"] / (viewer_entry["distance"] / 1000.0)
                viewer_summary = ClubLeaderboardSummary(
                    rank=viewer_entry["rank"],
                    distance=viewer_entry["distance"],
                    activity_count=viewer_entry["activity_count"],
                    longest_distance=viewer_entry["longest_distance"],
                    avg_pace=avg_pace,
                    elevation_gain=viewer_entry["elevation_gain"],
                )
            else:
                viewer_summary = ClubLeaderboardSummary(
                    rank=None,
                    distance=0.0,
                    activity_count=0,
                    longest_distance=0.0,
                    avg_pace=None,
                    elevation_gain=0.0,
                )

        return items, pagination, viewer_summary

    def recent_activity(
        self,
        db: Session,
        club_id: str,
        viewer_id: str | None,
        page: int,
        per_page: int,
    ) -> tuple[list[SummaryActivity], PaginatedResponseMetadata]:
        club = db.get(Club, club_id)
        if not club:
            raise NotFoundError("Club not found")
        if not self._can_read_club_content(db, club, viewer_id):
            raise ForbiddenError("Club membership required")

        member_ids = list(
            db.scalars(select(ClubMember.athlete_id).where(ClubMember.club_id == club_id)).all()
        )
        if not member_ids:
            return [], PaginatedResponseMetadata(page=page, per_page=per_page, total_items=0, total_pages=0)

        activities = db.scalars(
            select(Activity)
            .options(joinedload(Activity.metrics))
            .where(Activity.athlete_id.in_(member_ids))
            .order_by(Activity.start_date.desc())
        ).unique().all()

        visible = [a for a in activities if self._activity_visible(db, a, viewer_id)]
        page = max(1, page)
        per_page = min(max(1, per_page), 100)
        total_items = len(visible)
        total_pages = max(1, (total_items + per_page - 1) // per_page) if total_items else 0
        start_idx = (page - 1) * per_page
        page_slice = visible[start_idx : start_idx + per_page]

        items = [activity_service.to_summary(db, a, viewer_id) for a in page_slice]
        pagination = PaginatedResponseMetadata(
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
        )
        return items, pagination


club_service = ClubService()
