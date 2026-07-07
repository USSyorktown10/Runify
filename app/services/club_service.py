from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ForbiddenError
from app.models.athlete import Athlete
from app.models.social import Club, ClubInvite, ClubJoinRequest, ClubMember, ClubPost
from app.schemas.club import DetailedClub, SummaryClub
from app.schemas.social import Post
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

    def to_detailed(self, db: Session, club: Club) -> DetailedClub:
        admins = db.scalars(
            select(ClubMember.athlete_id).where(
                ClubMember.club_id == club.id, ClubMember.role.in_(["admin", "owner"])
            )
        ).all()
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
        return self.to_detailed(db, club)

    def _is_admin(self, db: Session, club_id: str, athlete_id: str) -> bool:
        member = db.scalar(
            select(ClubMember).where(ClubMember.club_id == club_id, ClubMember.athlete_id == athlete_id)
        )
        return member is not None and member.role in ("owner", "admin")

    def join(self, db: Session, club_id: str, athlete_id: str) -> tuple[bool, str | None]:
        club = db.get(Club, club_id)
        if not club:
            return False, "Club not found"
        existing = db.scalar(
            select(ClubMember).where(ClubMember.club_id == club_id, ClubMember.athlete_id == athlete_id)
        )
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
        db.add(ClubInvite(club_id=club_id, athlete_id=athlete_id, invited_by=inviter_id))
        notification_service.create(db, athlete_id, "club_invite", inviter_id, {"club_id": club_id})
        db.commit()
        return True, None

    def create_post(self, db: Session, club_id: str, author_id: str, title: str, body: str) -> Post:
        member = db.scalar(
            select(ClubMember).where(ClubMember.club_id == club_id, ClubMember.athlete_id == author_id)
        )
        if not member:
            raise ForbiddenError("Must be a club member")
        post = ClubPost(club_id=club_id, author_id=author_id, title=title, body=body)
        db.add(post)
        db.commit()
        db.refresh(post)
        author = db.get(Athlete, author_id)
        return Post(
            id=post.id,
            club_id=club_id,
            author=to_summary(author),
            title=title,
            body=body,
            created_at=post.created_at.isoformat(),
        )


club_service = ClubService()
