from datetime import date

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.models.athlete import Athlete, AthletePreferences, AthleteStats, PersonalRecord
from app.schemas.athlete import (
    AthletePreferences as AthletePreferencesSchema,
)
from app.schemas.athlete import (
    AthleteStats as AthleteStatsSchema,
)
from app.schemas.athlete import (
    DetailedAthlete,
    EmailNotificationsSettings,
    MeAthlete,
    PrivacySettings,
    SummaryAthlete,
)
from app.schemas.athlete import (
    PersonalRecord as PersonalRecordSchema,
)


def to_summary(athlete: Athlete) -> SummaryAthlete:
    return SummaryAthlete(
        id=athlete.id,
        first_name=athlete.first_name,
        last_name=athlete.last_name,
        profile_picture_url=athlete.profile_picture_url,
        city=athlete.city,
        state=athlete.state,
        country=athlete.country,
    )


def to_privacy(prefs: AthletePreferences) -> PrivacySettings:
    return PrivacySettings(
        profile_visibility=prefs.profile_visibility,
        activity_visibility=prefs.activity_visibility,
        biometrics_visibility=prefs.biometrics_visibility,
    )


class AthleteService:
    def get_me(self, db: Session, athlete: Athlete) -> MeAthlete:
        prefs = db.query(AthletePreferences).filter_by(athlete_id=athlete.id).first()
        from app.models.social import Integration as IntModel

        wearable = db.scalar(
            select(func.count()).select_from(IntModel).where(
                IntModel.athlete_id == athlete.id, IntModel.connected_at.isnot(None)
            )
        ) or 0
        return MeAthlete(
            id=athlete.id,
            username=athlete.username,
            email=athlete.email,
            first_name=athlete.first_name,
            last_name=athlete.last_name,
            city=athlete.city,
            state=athlete.state,
            country=athlete.country,
            profile_picture_url=athlete.profile_picture_url,
            gender=athlete.gender,
            birthdate=athlete.birthdate.isoformat() if athlete.birthdate else None,
            weight_kg=athlete.weight_kg,
            height_cm=athlete.height_cm,
            created=athlete.created_at.isoformat(),
            wearable_connected=wearable > 0,
            privacy_settings=to_privacy(prefs) if prefs else PrivacySettings(),
        )

    def get_detailed(self, db: Session, athlete_id: str, viewer: Athlete | None) -> DetailedAthlete:
        athlete = db.get(Athlete, athlete_id)
        if not athlete:
            raise NotFoundError("Athlete not found")
        prefs = db.query(AthletePreferences).filter_by(athlete_id=athlete_id).first()
        stats = db.query(AthleteStats).filter_by(athlete_id=athlete_id).first()
        records = db.scalars(select(PersonalRecord).where(PersonalRecord.athlete_id == athlete_id)).all()
        from app.models.social import Integration as IntModel
        wearable = db.scalar(
            select(func.count()).select_from(IntModel).where(IntModel.athlete_id == athlete_id, IntModel.connected_at.isnot(None))
        ) or 0
        return DetailedAthlete(
            id=athlete.id,
            username=athlete.username,
            first_name=athlete.first_name,
            last_name=athlete.last_name,
            city=athlete.city,
            state=athlete.state,
            country=athlete.country,
            profile_picture_url=athlete.profile_picture_url,
            created=athlete.created_at.isoformat(),
            wearable_connected=wearable > 0,
            stats=AthleteStatsSchema(
                current_ftp=stats.current_ftp if stats else 0,
                threshold_pace=stats.threshold_pace if stats else 3.5,
                ytd_run_totals=stats.ytd_run_totals if stats else 0,
                all_time_run_totals=stats.all_time_run_totals if stats else 0,
            ),
            privacy_settings=to_privacy(prefs) if prefs else PrivacySettings(),
            personal_records=[
                PersonalRecordSchema(
                    distance_name=r.distance_name,
                    time_in_seconds=r.time_in_seconds,
                    activity_id=r.activity_id or "",
                    achieved_date=r.achieved_date.isoformat(),
                )
                for r in records
            ],
        )

    def update_profile(self, db: Session, athlete: Athlete, data: dict) -> MeAthlete:
        for field in ("first_name", "last_name", "city", "state", "country", "profile_picture_url", "gender", "weight_kg", "height_cm"):
            if field in data and data[field] is not None:
                setattr(athlete, field, data[field])
        if data.get("birthdate"):
            athlete.birthdate = date.fromisoformat(data["birthdate"])
        db.commit()
        db.refresh(athlete)
        return self.get_me(db, athlete)

    def update_stats(self, db: Session, athlete_id: str, data: dict) -> AthleteStatsSchema:
        stats = db.query(AthleteStats).filter_by(athlete_id=athlete_id).first()
        if not stats:
            raise NotFoundError()
        if data.get("current_ftp") is not None:
            stats.current_ftp = data["current_ftp"]
        if data.get("threshold_pace") is not None:
            stats.threshold_pace = data["threshold_pace"]
        db.commit()
        return AthleteStatsSchema(
            current_ftp=stats.current_ftp,
            threshold_pace=stats.threshold_pace,
            ytd_run_totals=stats.ytd_run_totals,
            all_time_run_totals=stats.all_time_run_totals,
        )

    def get_preferences(self, db: Session, athlete_id: str) -> AthletePreferencesSchema:
        prefs = db.query(AthletePreferences).filter_by(athlete_id=athlete_id).first()
        if not prefs:
            raise NotFoundError()
        return AthletePreferencesSchema(
            measurement_system=prefs.measurement_system,
            privacy_settings=to_privacy(prefs),
            theme=prefs.theme,
            email_notifications=EmailNotificationsSettings(
                comments=prefs.email_comments,
                likes=prefs.email_likes,
                follow_requests=prefs.email_follow_requests,
                club_invites=prefs.email_club_invites,
            ),
        )

    def update_preferences(self, db: Session, athlete_id: str, data: dict) -> tuple[bool, str | None]:
        prefs = db.query(AthletePreferences).filter_by(athlete_id=athlete_id).first()
        if not prefs:
            return False, "Preferences not found"
        if data.get("measurement_system"):
            prefs.measurement_system = data["measurement_system"]
        if data.get("theme"):
            prefs.theme = data["theme"]
        ps = data.get("privacy_settings")
        if ps:
            prefs.profile_visibility = ps.get("profile_visibility", prefs.profile_visibility)
            prefs.activity_visibility = ps.get("activity_visibility", prefs.activity_visibility)
            prefs.biometrics_visibility = ps.get("biometrics_visibility", prefs.biometrics_visibility)
        en = data.get("email_notifications")
        if en:
            prefs.email_comments = en.get("comments", prefs.email_comments)
            prefs.email_likes = en.get("likes", prefs.email_likes)
            prefs.email_follow_requests = en.get("follow_requests", prefs.email_follow_requests)
            prefs.email_club_invites = en.get("club_invites", prefs.email_club_invites)
        db.commit()
        return True, None

    def search(self, db: Session, query: str, viewer: Athlete, page: int, per_page: int):
        stmt = select(Athlete).where(
            or_(
                Athlete.username.ilike(f"%{query}%"),
                Athlete.first_name.ilike(f"%{query}%"),
                Athlete.last_name.ilike(f"%{query}%"),
                Athlete.city.ilike(f"%{query}%"),
            )
        )
        return stmt


athlete_service = AthleteService()
