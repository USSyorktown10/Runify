from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.pagination import decode_datetime_cursor, encode_datetime_cursor
from app.models.activity import Activity
from app.models.athlete import Athlete
from app.models.social import AthletePost, ClubPost, Follow
from app.schemas.social import FeedItem
from app.services.activity_service import activity_service
from app.services.athlete_service import to_summary


class FeedService:
    def get_feed(
        self,
        db: Session,
        athlete_ids: list[str],
        cursor: str | None,
        limit: int,
        viewer_id: str,
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

        club_posts = db.scalars(
            select(ClubPost).order_by(ClubPost.created_at.desc()).limit(limit * 2)
        ).all()
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
                            is_liked=False,
                        ),
                    )
                )
            elif item_type == "club_post":
                from app.schemas.social import Post
                author = db.get(Athlete, obj.author_id)
                feed.append(
                    FeedItem(
                        id=obj.id,
                        type="club_post",
                        athlete=to_summary(author),
                        created_at=created,
                        club_post_data=Post(
                            id=obj.id,
                            club_id=obj.club_id,
                            author=to_summary(author),
                            title=obj.title,
                            body=obj.body,
                            created_at=created,
                        ),
                    )
                )

        next_cursor = encode_datetime_cursor(page_items[-1][3]) if len(page_items) == limit and page_items else None
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
        return self.get_feed(db, following_ids, cursor, limit, viewer.id)


feed_service = FeedService()
