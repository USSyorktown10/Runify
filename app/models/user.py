from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, 
    Date, 
    Float, 
    Boolean, 
    DateTime, 
    func, 
    ForeignKey,
    Integer
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.mutable import MutableList
from datetime import date, datetime
from app.db.base import Base, new_uuid
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from app.models.auth_session import AuthSession

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid())
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    first_name: Mapped[str] = mapped_column(String(100), default="")
    last_name: Mapped[str] = mapped_column(String(100), default="")
    city: Mapped[str] = mapped_column(String(100), default="")
    state: Mapped[str] = mapped_column(String(100), default="")
    country: Mapped[str] = mapped_column(String(100), default="")
    profile_picture_url: Mapped[str] = mapped_column(String(512), default="")
    bio: Mapped[str] = mapped_column(String(200), default="")
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    birthdate: Mapped[date | None] = mapped_column(Date, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    preferences: Mapped["UserPreferences"] = relationship(back_populates="user", uselist=False)
    stats: Mapped["UserStats"] = relationship(back_populates="user", uselist=False)
    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    
class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid())
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    measurement_system: Mapped[str] = mapped_column(String(16), default="metric")
    profile_visibility: Mapped[str] = mapped_column(String(16), default="public")
    activity_visibility: Mapped[str] = mapped_column(String(16), default="followers")
    metrics_visibility: Mapped[str] = mapped_column(String(16), default="followers")
    map_visibility: Mapped[str] = mapped_column(String(16), default="followers")
    request_to_follow: Mapped[bool] = mapped_column(Boolean, default=True)
    theme: Mapped[str] = mapped_column(String(16), default="system")
    email_comments: Mapped[bool] = mapped_column(Boolean, default=True)
    email_likes: Mapped[bool] = mapped_column(Boolean, default=True)
    email_follow_requests: Mapped[bool] = mapped_column(Boolean, default=True)
    email_club_invites: Mapped[bool] = mapped_column(Boolean, default=True)
    
    user: Mapped["User"] = relationship(back_populates="preferences")

class UserStats(Base):
    __tablename__ = "user_stats"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid())
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    current_ftp: Mapped[int] = mapped_column(Integer, default=0)
    threshold_pace: Mapped[float] = mapped_column(Float, default=4.5)
    four_week_run_totals: Mapped[list[float]] = mapped_column(MutableList.as_mutable(ARRAY(Float)), default=list)
    ytd_run_totals: Mapped[float] = mapped_column(Float, default=0.0)
    all_time_run_totals: Mapped[float] = mapped_column(Float, default=0.0)
    
    user: Mapped["User"] = relationship(back_populates="stats")
    
class PersonalRecord(Base):
    __tablename__ = "personal_records"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid())
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    distance_name: Mapped[str] = mapped_column(String(32))
    distance_meters: Mapped[float] = mapped_column(Float)
    time_seconds: Mapped[int] = mapped_column(Integer)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="SET NULL"), nullable=True)
    achieved_date: Mapped[datetime] = mapped_column(DateTime(timezone=True)) 