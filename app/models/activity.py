from datetime import datetime, timezone
from sqlalchemy import (
    DateTime, 
    ForeignKey, 
    String, 
    Text, 
    Float, 
    Integer,
    DateTime,
    Boolean
)
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class Activity(Base):
    __tablename__ = "activites"
    
    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
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
    percieved_effort: Mapped[int | None] = mapped_column(Integer, nullable=True)
    visibility: Mapped[str] = mapped_column(String(16), default="followers")
    metrics_visiblity: Mapped[str] = mapped_column(String(16), default="followers")
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    raw_file_path: Mapped[str] = mapped_column(String(512), nullable=True)