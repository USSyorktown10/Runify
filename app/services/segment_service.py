
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.models.activity import Activity, ActivityStream
from app.models.athlete import Athlete
from app.models.segment import Segment, SegmentEffort, SegmentStar
from app.schemas.segment import DetailedSegment, SummarySegment

TOLERANCE_M = 50.0


def age_group_for(birthdate) -> str | None:
    if not birthdate:
        return None
    from datetime import date
    age = (date.today() - birthdate).days // 365
    if age < 18:
        return "under_18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def weight_class_for(weight_kg: float | None) -> str | None:
    if weight_kg is None:
        return None
    if weight_kg < 60:
        return "light"
    if weight_kg < 75:
        return "medium"
    if weight_kg < 90:
        return "heavy"
    return "super_heavy"


class SegmentService:
    def _is_starred(self, db: Session, segment_id: str, athlete_id: str | None) -> bool:
        if not athlete_id:
            return False
        return db.scalar(
            select(SegmentStar).where(SegmentStar.segment_id == segment_id, SegmentStar.athlete_id == athlete_id)
        ) is not None

    def to_summary(self, db: Session, seg: Segment, viewer_id: str | None) -> SummarySegment:
        return SummarySegment(
            id=seg.id,
            name=seg.name,
            activity_type=seg.activity_type,
            distance=seg.distance,
            average_grade=seg.average_grade,
            start_latlng=[seg.start_lat, seg.start_lng],
            end_latlng=[seg.end_lat, seg.end_lng],
            is_starred=self._is_starred(db, seg.id, viewer_id),
        )

    def to_detailed(self, db: Session, seg: Segment, viewer_id: str | None) -> DetailedSegment:
        return DetailedSegment(
            id=seg.id,
            name=seg.name,
            activity_type=seg.activity_type,
            distance=seg.distance,
            average_grade=seg.average_grade,
            start_latlng=[seg.start_lat, seg.start_lng],
            end_latlng=[seg.end_lat, seg.end_lng],
            is_starred=self._is_starred(db, seg.id, viewer_id),
            polyline=seg.polyline,
            elevation_high=seg.elevation_high,
            elevation_low=seg.elevation_low,
            total_effort_count=seg.total_effort_count,
            total_athlete_count=seg.total_athlete_count,
            star_count=seg.star_count,
        )

    def create_from_activity(
        self, db: Session, athlete_id: str, activity_id: str, start_index: int, end_index: int, name: str
    ) -> DetailedSegment:
        activity = db.get(Activity, activity_id)
        if not activity or activity.athlete_id != athlete_id:
            raise NotFoundError("Activity not found")
        lat_stream = db.scalar(
            select(ActivityStream).where(
                ActivityStream.activity_id == activity_id,
                ActivityStream.metric_key == "lat",
                ActivityStream.resolution == "high",
            )
        )
        lng_stream = db.scalar(
            select(ActivityStream).where(
                ActivityStream.activity_id == activity_id,
                ActivityStream.metric_key == "lng",
                ActivityStream.resolution == "high",
            )
        )
        if not lat_stream or not lng_stream:
            raise NotFoundError("Activity has no GPS data")
        lats, lngs = lat_stream.data, lng_stream.data
        start_lat, start_lng = lats[start_index], lngs[start_index]
        end_lat, end_lng = lats[end_index], lngs[end_index]
        dist = activity.distance * (end_index - start_index) / max(len(lats), 1)
        seg = Segment(
            name=name,
            distance=dist,
            start_lat=start_lat,
            start_lng=start_lng,
            end_lat=end_lat,
            end_lng=end_lng,
            polyline=activity.polyline,
            creator_id=athlete_id,
            source_activity_id=activity_id,
            start_index=start_index,
            end_index=end_index,
        )
        db.add(seg)
        db.commit()
        db.refresh(seg)
        return self.to_detailed(db, seg, athlete_id)

    def search(self, db: Session, query: str | None, activity_type: str | None):
        stmt = select(Segment)
        if query:
            stmt = stmt.where(Segment.name.ilike(f"%{query}%"))
        if activity_type:
            stmt = stmt.where(Segment.activity_type == activity_type)
        return stmt.order_by(Segment.created_at.desc())

    def star(self, db: Session, segment_id: str, athlete_id: str) -> tuple[bool, str | None]:
        seg = db.get(Segment, segment_id)
        if not seg:
            return False, "Segment not found"
        existing = db.scalar(
            select(SegmentStar).where(SegmentStar.segment_id == segment_id, SegmentStar.athlete_id == athlete_id)
        )
        if not existing:
            db.add(SegmentStar(segment_id=segment_id, athlete_id=athlete_id))
            seg.star_count += 1
            db.commit()
        return True, None

    def unstar(self, db: Session, segment_id: str, athlete_id: str) -> tuple[bool, str | None]:
        star = db.scalar(
            select(SegmentStar).where(SegmentStar.segment_id == segment_id, SegmentStar.athlete_id == athlete_id)
        )
        if star:
            seg = db.get(Segment, segment_id)
            if seg and seg.star_count > 0:
                seg.star_count -= 1
            db.delete(star)
            db.commit()
        return True, None

    def match_activity(self, db: Session, activity: Activity) -> None:
        lat_stream = db.scalar(
            select(ActivityStream).where(
                ActivityStream.activity_id == activity.id,
                ActivityStream.metric_key == "lat",
                ActivityStream.resolution == "high",
            )
        )
        lng_stream = db.scalar(
            select(ActivityStream).where(
                ActivityStream.activity_id == activity.id,
                ActivityStream.metric_key == "lng",
                ActivityStream.resolution == "high",
            )
        )
        if not lat_stream or not lng_stream or not lat_stream.data:
            return
        lats, lngs = lat_stream.data, lng_stream.data
        min_lat, max_lat = min(lats), max(lats)
        min_lng, max_lng = min(lngs), max(lngs)
        pad = 0.001
        candidates = db.scalars(
            select(Segment).where(
                Segment.start_lat.between(min_lat - pad, max_lat + pad),
                Segment.start_lng.between(min_lng - pad, max_lng + pad),
            )
        ).all()
        for seg in candidates:
            start_idx = self._find_nearest(lats, lngs, seg.start_lat, seg.start_lng)
            end_idx = self._find_nearest(lats, lngs, seg.end_lat, seg.end_lng)
            if start_idx is None or end_idx is None or end_idx <= start_idx:
                continue
            elapsed = int(activity.moving_time * (end_idx - start_idx) / max(len(lats), 1))
            effort = SegmentEffort(
                segment_id=seg.id,
                activity_id=activity.id,
                athlete_id=activity.athlete_id,
                elapsed_time=elapsed,
                moving_time=elapsed,
                start_date=activity.start_date,
            )
            db.add(effort)
            seg.total_effort_count += 1
            existing_athletes = db.scalar(
                select(func.count(func.distinct(SegmentEffort.athlete_id))).where(
                    SegmentEffort.segment_id == seg.id
                )
            )
            seg.total_athlete_count = existing_athletes or 1

    def _find_nearest(self, lats: list, lngs: list, target_lat: float, target_lng: float) -> int | None:
        from app.services.metrics_engine.formulas import haversine
        best_idx, best_dist = None, TOLERANCE_M
        for i, (la, ln) in enumerate(zip(lats, lngs, strict=False)):
            d = haversine(la, ln, target_lat, target_lng)
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx

    def leaderboard(
        self,
        db: Session,
        segment_id: str,
        gender: str | None,
        age_group: str | None,
        weight_class: str | None,
        page: int,
        per_page: int,
    ):
        stmt = (
            select(SegmentEffort, Athlete)
            .join(Athlete, Athlete.id == SegmentEffort.athlete_id)
            .where(SegmentEffort.segment_id == segment_id)
            .order_by(SegmentEffort.elapsed_time.asc())
        )
        efforts = db.execute(stmt).all()
        filtered = []
        for effort, athlete in efforts:
            if gender and athlete.gender != gender:
                continue
            if age_group and age_group_for(athlete.birthdate) != age_group:
                continue
            if weight_class and weight_class_for(athlete.weight_kg) != weight_class:
                continue
            filtered.append((effort, athlete))
        return filtered


segment_service = SegmentService()
