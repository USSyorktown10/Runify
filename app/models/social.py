from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, relationship

from app.db.base import Base
from app.models.athlete import new_uuid


class Follow(Base):
    __tablename__ = "follows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    follower_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    following_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="following")  # following, pending
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Block(Base):
    __tablename__ = "blocks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    blocker_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    blocked_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    author_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    target_type: Mapped[str] = mapped_column(String(16))  # activity, post
    target_id: Mapped[str] = mapped_column(String(36), index=True)
    text: Mapped[str] = mapped_column(Text)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Like(Base):
    __tablename__ = "likes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    target_type: Mapped[str] = mapped_column(String(16))  # activity, post, comment
    target_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AthletePost(Base):
    __tablename__ = "athlete_posts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    media_urls: Mapped[list] = mapped_column(JSON, default=list)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class Club(Base):
    __tablename__ = "clubs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    profile_picture_url: Mapped[str] = mapped_column(String(512), default="")
    cover_photo_url: Mapped[str] = mapped_column(String(512), default="")
    is_private: Mapped[bool] = mapped_column(Boolean, default=False)
    creator_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"))
    tags: Mapped[list] = mapped_column(JSON, default=list)
    member_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClubMember(Base):
    __tablename__ = "club_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    club_id: Mapped[str] = mapped_column(ForeignKey("clubs.id", ondelete="CASCADE"), index=True)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="member")
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClubInvite(Base):
    __tablename__ = "club_invites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    club_id: Mapped[str] = mapped_column(ForeignKey("clubs.id", ondelete="CASCADE"), index=True)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    invited_by: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClubJoinRequest(Base):
    __tablename__ = "club_join_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    club_id: Mapped[str] = mapped_column(ForeignKey("clubs.id", ondelete="CASCADE"), index=True)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClubPost(Base):
    __tablename__ = "club_posts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    club_id: Mapped[str] = mapped_column(ForeignKey("clubs.id", ondelete="CASCADE"), index=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    club: Mapped["Club"] = relationship("Club", foreign_keys=[club_id])


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(32))
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    sender_id: Mapped[str | None] = mapped_column(ForeignKey("athletes.id", ondelete="SET NULL"), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class Integration(Base):
    __tablename__ = "integrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    external_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    access_token: Mapped[str | None] = mapped_column(String(512), nullable=True)
    refresh_token: Mapped[str | None] = mapped_column(String(512), nullable=True)
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    oauth_state: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    reporter_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"))
    target_type: Mapped[str] = mapped_column(String(16))  # activity, athlete, club
    target_id: Mapped[str] = mapped_column(String(36))
    reason: Mapped[str] = mapped_column(String(128))
    details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
