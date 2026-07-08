from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.errors import NotFoundError
from app.models.activity import (
    Activity,
    ActivityMetric,
    ActivitySplit,
    ActivityStream,
    Upload,
)
from app.models.athlete import AthletePreferences, AthleteStats
from app.models.social import Like
from app.schemas.activity import (
    DetailedActivity,
    Lap,
    PowerCurve,
    PowerCurveValue,
    Split,
    Stream,
    SummaryActivity,
)
from app.schemas.common import (
    DynamicActivityZone,
    DynamicMetricDistribution,
    DynamicWorkoutMetric,
)
from app.services.metrics_engine.formulas import power_curve as compute_power_curve
from app.services.metrics_engine.pipeline import process_upload


class ActivityService:
    def _is_liked(self, db: Session, athlete_id: str | None, activity_id: str) -> bool:
        if not athlete_id:
            return False
        return db.scalar(
            select(Like).where(
                Like.athlete_id == athlete_id,
                Like.target_type == "activity",
                Like.target_id == activity_id,
            )
        ) is not None

    def _summary_metrics(self, metrics: list[ActivityMetric]) -> list[DynamicWorkoutMetric]:
        common = {"distance", "avg_speed", "avg_hr", "elevation_gain", "vo2_max"}
        return [
            DynamicWorkoutMetric(
                key=m.key, value=m.value, source=m.source, unit=m.unit, display_name=m.display_name
            )
            for m in metrics
            if m.key in common
        ]

    def to_summary(self, db: Session, activity: Activity, viewer_id: str | None) -> SummaryActivity:
        return SummaryActivity(
            id=activity.id,
            athlete_id=activity.athlete_id,
            name=activity.name,
            activity_type=activity.activity_type,
            distance=activity.distance,
            moving_time=activity.moving_time,
            start_date=activity.start_date.isoformat(),
            polyline_summary=activity.polyline_summary,
            device_name=activity.device_name,
            visibility=activity.visibility,
            biometrics_visibility=activity.biometrics_visibility,
            like_count=activity.like_count,
            comment_count=activity.comment_count,
            is_liked=self._is_liked(db, viewer_id, activity.id),
            metrics=self._summary_metrics(activity.metrics),
        )

    def to_detailed(self, db: Session, activity: Activity, viewer_id: str | None) -> DetailedActivity:
        return DetailedActivity(
            id=activity.id,
            athlete_id=activity.athlete_id,
            name=activity.name,
            description=activity.description,
            activity_type=activity.activity_type,
            distance=activity.distance,
            moving_time=activity.moving_time,
            elapsed_time=activity.elapsed_time,
            start_date=activity.start_date.isoformat(),
            polyline=activity.polyline,
            device_name=activity.device_name,
            gear_id=activity.gear_id,
            perceived_exertion=activity.perceived_exertion,
            visibility=activity.visibility,
            biometrics_visibility=activity.biometrics_visibility,
            like_count=activity.like_count,
            comment_count=activity.comment_count,
            is_liked=self._is_liked(db, viewer_id, activity.id),
            metrics=[
                DynamicWorkoutMetric(key=m.key, value=m.value, source=m.source, unit=m.unit, display_name=m.display_name)
                for m in activity.metrics
            ],
            distributions=[
                DynamicMetricDistribution(key=d.key, display_name=d.display_name, unit=d.unit, buckets=d.buckets)
                for d in activity.distributions
            ],
            zones=[
                DynamicActivityZone(
                    key=z.key,
                    display_name=z.display_name,
                    unit=z.unit,
                    reference_value=z.reference_value,
                    reference_name=z.reference_name,
                    zones=z.zones,
                )
                for z in activity.zones
            ],
            laps=[
                Lap(
                    id=lap.id,
                    lap_index=lap.lap_index,
                    name=lap.name,
                    start_date=lap.start_date.isoformat(),
                    elapsed_time=lap.elapsed_time,
                    moving_time=lap.moving_time,
                    distance=lap.distance,
                    average_speed=lap.average_speed,
                )
                for lap in activity.laps
            ],
        )

    def list_activities(self, db: Session, athlete_id: str, page: int, per_page: int, sort_order: str):
        stmt = (
            select(Activity)
            .options(joinedload(Activity.metrics))
            .where(Activity.athlete_id == athlete_id)
        )
        if sort_order == "oldest":
            stmt = stmt.order_by(Activity.start_date.asc())
        else:
            stmt = stmt.order_by(Activity.start_date.desc())
        return stmt

    def get_activity(self, db: Session, activity_id: str) -> Activity:
        activity = db.scalar(
            select(Activity)
            .options(
                joinedload(Activity.metrics),
                joinedload(Activity.distributions),
                joinedload(Activity.zones),
                joinedload(Activity.laps),
            )
            .where(Activity.id == activity_id)
        )
        if not activity:
            raise NotFoundError("Activity not found")
        return activity

    def create_manual(self, db: Session, athlete_id: str, data: dict) -> tuple[bool, str | None, str | None]:
        prefs = db.query(AthletePreferences).filter_by(athlete_id=athlete_id).first()
        activity = Activity(
            athlete_id=athlete_id,
            name=data["name"],
            activity_type=data.get("activity_type", "run"),
            start_date=datetime.fromisoformat(data["start_date"]),
            elapsed_time=data["elapsed_time"],
            moving_time=data["elapsed_time"],
            distance=data["distance"],
            description=data.get("description", ""),
            perceived_exertion=data.get("perceived_exertion"),
            gear_id=data.get("gear_id"),
            visibility=data.get("visibility") or (prefs.activity_visibility if prefs else "followers"),
            biometrics_visibility=data.get("biometrics_visibility") or (prefs.biometrics_visibility if prefs else "followers"),
        )
        db.add(activity)
        db.flush()
        for m in data.get("metrics", []):
            db.add(
                ActivityMetric(
                    activity_id=activity.id,
                    key=m["key"] if isinstance(m, dict) else m.key,
                    value=m["value"] if isinstance(m, dict) else m.value,
                    source=m.get("source", "raw") if isinstance(m, dict) else m.source,
                    unit=m.get("unit", "") if isinstance(m, dict) else m.unit,
                    display_name=m.get("display_name", m["key"]) if isinstance(m, dict) else m.display_name,
                )
            )
        db.commit()
        return True, activity.id, None

    def update(self, db: Session, activity_id: str, athlete_id: str, data: dict) -> tuple[bool, str | None]:
        activity = self.get_activity(db, activity_id)
        if activity.athlete_id != athlete_id:
            return False, "Forbidden"
        for field in ("name", "description", "gear_id", "visibility", "biometrics_visibility"):
            if field in data and data[field] is not None:
                setattr(activity, field, data[field])
        db.commit()
        return True, None

    def delete(self, db: Session, activity_id: str, athlete_id: str) -> tuple[bool, str | None]:
        activity = self.get_activity(db, activity_id)
        if activity.athlete_id != athlete_id:
            return False, "Forbidden"
        db.delete(activity)
        db.commit()
        return True, None

    def get_streams(self, db: Session, activity_id: str, keys: list[str], resolution: str) -> list[Stream]:
        stmt = select(ActivityStream).where(
            ActivityStream.activity_id == activity_id,
            ActivityStream.resolution == resolution,
        )
        if keys:
            stmt = stmt.where(ActivityStream.metric_key.in_(keys))
        streams = db.scalars(stmt).all()
        return [
            Stream(
                metric_key=s.metric_key,
                stream_type=s.stream_type,
                data=s.data,
                axis=s.axis,
                axis_type=s.axis_type,
                original_size=s.original_size,
                resolution=s.resolution,
            )
            for s in streams
        ]

    def get_splits(self, db: Session, activity_id: str) -> list[Split]:
        splits = db.scalars(
            select(ActivitySplit).where(ActivitySplit.activity_id == activity_id).order_by(ActivitySplit.index)
        ).all()
        return [
            Split(
                index=s.index,
                distance=s.distance,
                elapsed_time=s.elapsed_time,
                elevation_difference=s.elevation_difference,
                average_speed=s.average_speed,
            )
            for s in splits
        ]

    def get_power_curve(self, db: Session, activity_id: str) -> PowerCurve:
        stream = db.scalar(
            select(ActivityStream).where(
                ActivityStream.activity_id == activity_id,
                ActivityStream.metric_key == "power",
                ActivityStream.resolution == "high",
            )
        )
        if not stream or not stream.data:
            return PowerCurve(curve_values=[])
        values = compute_power_curve(stream.data)
        return PowerCurve(
            curve_values=[PowerCurveValue(**v) for v in values]
        )

    def process_upload_task(self, db: Session, upload_id: str) -> None:
        upload = db.get(Upload, upload_id)
        if not upload:
            return
        try:
            upload.status = "processing"
            db.commit()
            stats = db.query(AthleteStats).filter_by(athlete_id=upload.athlete_id).first()
            threshold = stats.threshold_pace if stats else 3.5
            activity = process_upload(db, upload, threshold_pace=threshold)
            from app.services.segment_service import segment_service
            segment_service.match_activity(db, activity)
            db.commit()
        except Exception as e:
            upload.status = "failed"
            upload.error_message = str(e)
            db.commit()


activity_service = ActivityService()
