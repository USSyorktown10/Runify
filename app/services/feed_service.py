from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.pagination import decode_datetime_cursor, encode_cursor
from app.models.activity import Activity
from app.models.athlete import Athlete
from app.models.social import AthletePost, ClubMember, ClubPost, Follow, Like
from app.schemas.social import FeedItem
from app.services.activity_service import activity_service
from app.services.athlete_service import to_summary
from app.services.club_service import club_service


class FeedService:
    def get_feed(
        self,
        db: Session,
        athlete_ids: list[str],
        cursor: str | None,
        limit: int,
        viewer_id: str,
        *,
        include_club_posts: bool = True,
        club_ids: list[str] | None = None,
    ) -> tuple[list[FeedItem], str | None]:
        following = athlete_ids
        items: list[tuple[str, str, object, str]] = []

        activities = db.scalars(
            select(Activity)
            .options(joinedload(Activity.metrics))
            .where(Activity.athlete_id.in_(following))
            .order_by(Activity.start_date.desc())
            .limit(limit * 2)
        ).unique().all()
        for a in activities:
            items.append(("activity", a.id, a, a.start_date.isoformat()))

        posts = db.scalars(
            select(AthletePost)
            .where(AthletePost.athlete_id.in_(following))
            .order_by(AthletePost.created_at.desc())
            .limit(limit * 2)
        ).all()
        for p in posts:
            items.append(("post", p.id, p, p.created_at.isoformat()))

        if include_club_posts and club_ids:
            club_posts = db.scalars(
                select(ClubPost)
                .options(joinedload(ClubPost.club))
                .where(ClubPost.club_id.in_(club_ids))
                .order_by(ClubPost.created_at.desc())
                .limit(limit * 2)
            ).unique().all()
            for cp in club_posts:
                items.append(("club_post", cp.id, cp, cp.created_at.isoformat()))

        items.sort(key=lambda x: x[3], reverse=True)
        cursor_dt = decode_datetime_cursor(cursor)
        if cursor_dt:
            items = [i for i in items if i[3] < cursor_dt.isoformat()]

        page_items = items[:limit]
        feed: list[FeedItem] = []
        for item_type, _id, obj, created in page_items:
            if item_type == "activity":
                athlete = db.get(Athlete, obj.athlete_id)
                feed.append(
                    FeedItem(
                        id=obj.id,
                        type="activity",
                        athlete=to_summary(athlete),
                        created_at=created,
                        activity_data=activity_service.to_summary(db, obj, viewer_id),
                    )
                )
            elif item_type == "post":
                athlete = db.get(Athlete, obj.athlete_id)
                from app.schemas.social import AthletePost as AthletePostSchema
                is_liked = db.scalar(
                    select(Like).where(
                        Like.athlete_id == viewer_id,
                        Like.target_type == "post",
                        Like.target_id == obj.id,
                    )
                ) is not None
                feed.append(
                    FeedItem(
                        id=obj.id,
                        type="post",
                        athlete=to_summary(athlete),
                        created_at=created,
                        post_data=AthletePostSchema(
                            id=obj.id,
                            athlete_id=obj.athlete_id,
                            text=obj.text,
                            media_urls=obj.media_urls,
                            created_at=created,
                            like_count=obj.like_count,
                            comment_count=obj.comment_count,
                            is_liked=is_liked,
                        ),
                    )
                )
            elif item_type == "club_post":
                author = db.get(Athlete, obj.author_id)
                feed.append(
                    FeedItem(
                        id=obj.id,
                        type="club_post",
                        athlete=to_summary(author),
                        created_at=created,
                        club_post_data=club_service.to_post(db, obj, viewer_id),
                    )
                )

        next_cursor = encode_cursor(page_items[-1][3]) if len(page_items) == limit and page_items else None
        return feed, next_cursor

    def get_home_feed(self, db: Session, viewer: Athlete, cursor: str | None, limit: int):
        following_ids = list(
            db.scalars(
                select(Follow.following_id).where(
                    Follow.follower_id == viewer.id, Follow.status == "following"
                )
            ).all()
        )
        following_ids.append(viewer.id)
        club_ids = list(
            db.scalars(select(ClubMember.club_id).where(ClubMember.athlete_id == viewer.id)).all()
        )
        return self.get_feed(
            db, following_ids, cursor, limit, viewer.id, include_club_posts=True, club_ids=club_ids
        )

    def get_athlete_profile_feed(
        self, db: Session, athlete_id: str, cursor: str | None, limit: int, viewer_id: str
    ):
        return self.get_feed(
            db, [athlete_id], cursor, limit, viewer_id, include_club_posts=False
        )


feed_service = FeedService()
