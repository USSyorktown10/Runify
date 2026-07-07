import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from app.models.auth import AuthSession

from app.db.base import Base


def new_uuid() -> str:
    return str(uuid.uuid4())


class Athlete(Base):
    __tablename__ = "athletes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    first_name: Mapped[str] = mapped_column(String(100), default="")
    last_name: Mapped[str] = mapped_column(String(100), default="")
    city: Mapped[str] = mapped_column(String(100), default="")
    state: Mapped[str] = mapped_column(String(100), default="")
    country: Mapped[str] = mapped_column(String(100), default="")
    profile_picture_url: Mapped[str] = mapped_column(String(512), default="")
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    birthdate: Mapped[date | None] = mapped_column(Date, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    preferences: Mapped["AthletePreferences"] = relationship(back_populates="athlete", uselist=False)
    stats: Mapped["AthleteStats"] = relationship(back_populates="athlete", uselist=False)
    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="athlete")


class AthletePreferences(Base):
    __tablename__ = "athlete_preferences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), unique=True)
    measurement_system: Mapped[str] = mapped_column(String(16), default="metric")
    profile_visibility: Mapped[str] = mapped_column(String(16), default="public")
    activity_visibility: Mapped[str] = mapped_column(String(16), default="followers")
    biometrics_visibility: Mapped[str] = mapped_column(String(16), default="followers")
    theme: Mapped[str] = mapped_column(String(16), default="system")
    email_comments: Mapped[bool] = mapped_column(Boolean, default=True)
    email_likes: Mapped[bool] = mapped_column(Boolean, default=True)
    email_follow_requests: Mapped[bool] = mapped_column(Boolean, default=True)
    email_club_invites: Mapped[bool] = mapped_column(Boolean, default=True)

    athlete: Mapped["Athlete"] = relationship(back_populates="preferences")


class AthleteStats(Base):
    __tablename__ = "athlete_stats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), unique=True)
    current_ftp: Mapped[int] = mapped_column(Integer, default=0)
    threshold_pace: Mapped[float] = mapped_column(Float, default=3.5)
    ytd_run_totals: Mapped[float] = mapped_column(Float, default=0.0)
    all_time_run_totals: Mapped[float] = mapped_column(Float, default=0.0)

    athlete: Mapped["Athlete"] = relationship(back_populates="stats")


class PersonalRecord(Base):
    __tablename__ = "personal_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    distance_name: Mapped[str] = mapped_column(String(32))
    distance_meters: Mapped[float] = mapped_column(Float)
    time_in_seconds: Mapped[int] = mapped_column(Integer)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="SET NULL"), nullable=True)
    achieved_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AthleteStream(Base):
    __tablename__ = "athlete_streams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    metric_key: Mapped[str] = mapped_column(String(64), index=True)
    stream_type: Mapped[str] = mapped_column(String(16), default="calculated")
    axis_type: Mapped[str] = mapped_column(String(16), default="time")
    resolution: Mapped[str] = mapped_column(String(16), default="high")
    original_size: Mapped[int] = mapped_column(Integer, default=0)
    data: Mapped[list] = mapped_column(JSON, default=list)
    axis: Mapped[list] = mapped_column(JSON, default=list)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
