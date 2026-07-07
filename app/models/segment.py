from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.athlete import new_uuid


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255))
    activity_type: Mapped[str] = mapped_column(String(32), default="run")
    distance: Mapped[float] = mapped_column(Float)
    average_grade: Mapped[float] = mapped_column(Float, default=0.0)
    start_lat: Mapped[float] = mapped_column(Float)
    start_lng: Mapped[float] = mapped_column(Float)
    end_lat: Mapped[float] = mapped_column(Float)
    end_lng: Mapped[float] = mapped_column(Float)
    polyline: Mapped[str] = mapped_column(Text, default="")
    elevation_high: Mapped[float] = mapped_column(Float, default=0.0)
    elevation_low: Mapped[float] = mapped_column(Float, default=0.0)
    creator_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"))
    source_activity_id: Mapped[str | None] = mapped_column(ForeignKey("activities.id", ondelete="SET NULL"), nullable=True)
    start_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_effort_count: Mapped[int] = mapped_column(Integer, default=0)
    total_athlete_count: Mapped[int] = mapped_column(Integer, default=0)
    star_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SegmentStar(Base):
    __tablename__ = "segment_stars"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"), index=True)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SegmentEffort(Base):
    __tablename__ = "segment_efforts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"), index=True)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    elapsed_time: Mapped[int] = mapped_column(Integer)
    moving_time: Mapped[int] = mapped_column(Integer)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    average_heartrate: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_power: Mapped[float | None] = mapped_column(Float, nullable=True)


class Route(Base):
    __tablename__ = "routes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    activity_type: Mapped[str] = mapped_column(String(32), default="run")
    distance: Mapped[float] = mapped_column(Float, default=0.0)
    elevation_gain: Mapped[float] = mapped_column(Float, default=0.0)
    polyline: Mapped[str] = mapped_column(Text, default="")
    polyline_summary: Mapped[str] = mapped_column(Text, default="")
    is_private: Mapped[bool] = mapped_column(Boolean, default=False)
    estimated_duration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RouteWaypoint(Base):
    __tablename__ = "route_waypoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.id", ondelete="CASCADE"), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    elevation: Mapped[float] = mapped_column(Float, default=0.0)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
