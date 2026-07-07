from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.athlete import new_uuid


class Gear(Base):
    __tablename__ = "gear"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    brand_name: Mapped[str] = mapped_column(String(128), default="")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    gear_type: Mapped[str] = mapped_column(String(32), default="shoe")
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    max_mileage: Mapped[float] = mapped_column(Float, default=0.0)
    is_retired: Mapped[bool] = mapped_column(Boolean, default=False)
    initial_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Activity(Base):
    __tablename__ = "activities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), default="Untitled Run")
    description: Mapped[str] = mapped_column(Text, default="")
    activity_type: Mapped[str] = mapped_column(String(32), default="run")
    distance: Mapped[float] = mapped_column(Float, default=0.0)
    moving_time: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_time: Mapped[int] = mapped_column(Integer, default=0)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    polyline: Mapped[str] = mapped_column(Text, default="")
    polyline_summary: Mapped[str] = mapped_column(Text, default="")
    device_name: Mapped[str] = mapped_column(String(128), default="")
    gear_id: Mapped[str | None] = mapped_column(ForeignKey("gear.id", ondelete="SET NULL"), nullable=True)
    perceived_exertion: Mapped[int | None] = mapped_column(Integer, nullable=True)
    visibility: Mapped[str] = mapped_column(String(16), default="followers")
    biometrics_visibility: Mapped[str] = mapped_column(String(16), default="followers")
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    raw_file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    metrics: Mapped[list["ActivityMetric"]] = relationship(back_populates="activity", cascade="all, delete-orphan")
    distributions: Mapped[list["ActivityDistribution"]] = relationship(back_populates="activity", cascade="all, delete-orphan")
    zones: Mapped[list["ActivityZone"]] = relationship(back_populates="activity", cascade="all, delete-orphan")
    streams: Mapped[list["ActivityStream"]] = relationship(back_populates="activity", cascade="all, delete-orphan")
    laps: Mapped[list["ActivityLap"]] = relationship(back_populates="activity", cascade="all, delete-orphan")
    splits: Mapped[list["ActivitySplit"]] = relationship(back_populates="activity", cascade="all, delete-orphan")


class ActivityMetric(Base):
    __tablename__ = "activity_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))
    unit: Mapped[str] = mapped_column(String(32))
    display_name: Mapped[str] = mapped_column(String(128))

    activity: Mapped["Activity"] = relationship(back_populates="metrics")


class ActivityDistribution(Base):
    __tablename__ = "activity_distributions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(128))
    unit: Mapped[str] = mapped_column(String(32))
    buckets: Mapped[list] = mapped_column(JSON, default=list)

    activity: Mapped["Activity"] = relationship(back_populates="distributions")


class ActivityZone(Base):
    __tablename__ = "activity_zones"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(128))
    unit: Mapped[str] = mapped_column(String(32))
    reference_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    zones: Mapped[list] = mapped_column(JSON, default=list)

    activity: Mapped["Activity"] = relationship(back_populates="zones")


class ActivityStream(Base):
    __tablename__ = "activity_streams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    metric_key: Mapped[str] = mapped_column(String(64), index=True)
    stream_type: Mapped[str] = mapped_column(String(16))
    axis_type: Mapped[str] = mapped_column(String(16))
    resolution: Mapped[str] = mapped_column(String(16), default="high")
    original_size: Mapped[int] = mapped_column(Integer, default=0)
    data: Mapped[list] = mapped_column(JSON, default=list)
    axis: Mapped[list] = mapped_column(JSON, default=list)

    activity: Mapped["Activity"] = relationship(back_populates="streams")


class ActivityLap(Base):
    __tablename__ = "activity_laps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    lap_index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(128), default="")
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    elapsed_time: Mapped[int] = mapped_column(Integer)
    moving_time: Mapped[int] = mapped_column(Integer)
    distance: Mapped[float] = mapped_column(Float)
    average_speed: Mapped[float] = mapped_column(Float)

    activity: Mapped["Activity"] = relationship(back_populates="laps")


class ActivitySplit(Base):
    __tablename__ = "activity_splits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    activity_id: Mapped[str] = mapped_column(ForeignKey("activities.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    distance: Mapped[float] = mapped_column(Float)
    elapsed_time: Mapped[int] = mapped_column(Integer)
    elevation_difference: Mapped[float] = mapped_column(Float, default=0.0)
    average_speed: Mapped[float] = mapped_column(Float)

    activity: Mapped["Activity"] = relationship(back_populates="splits")


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str] = mapped_column(String(512))
    file_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    activity_id: Mapped[str | None] = mapped_column(ForeignKey("activities.id", ondelete="SET NULL"), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
