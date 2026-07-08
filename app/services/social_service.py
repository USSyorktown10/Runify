from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError
from app.models.athlete import Athlete, AthletePreferences
from app.models.social import AthletePost, Block, ClubPost, Comment, Follow, Like
from app.schemas.social import AthletePost as AthletePostSchema
from app.schemas.social import Comment as CommentSchema
from app.services.athlete_service import to_summary
from app.services.notification_service import notification_service


class SocialService:
    def relationship_status(self, db: Session, viewer_id: str, target_id: str) -> str:
        if viewer_id == target_id:
            return "none"
        blocked = db.scalar(
            select(Block).where(
                or_(
                    (Block.blocker_id == viewer_id) & (Block.blocked_id == target_id),
                    (Block.blocker_id == target_id) & (Block.blocked_id == viewer_id),
                )
            )
        )
        if blocked:
            return "blocked"
        follow = db.scalar(
            select(Follow).where(Follow.follower_id == viewer_id, Follow.following_id == target_id)
        )
        if follow:
            return follow.status
        reverse = db.scalar(
            select(Follow).where(Follow.follower_id == target_id, Follow.following_id == viewer_id)
        )
        if reverse and reverse.status == "following":
            return "following"
        return "none"

    def follow(self, db: Session, follower_id: str, following_id: str) -> tuple[str, bool]:
        if follower_id == following_id:
            raise ConflictError("Cannot follow yourself")
        target_prefs = db.query(AthletePreferences).filter_by(athlete_id=following_id).first()
        status = "pending" if target_prefs and target_prefs.profile_visibility == "private" else "following"
        existing = db.scalar(
            select(Follow).where(Follow.follower_id == follower_id, Follow.following_id == following_id)
        )
        if existing:
            return existing.status, True
        db.add(Follow(follower_id=follower_id, following_id=following_id, status=status))
        if status == "pending":
            notification_service.create(
                db, following_id, "follow_request", follower_id, {"follower_id": follower_id}
            )
        db.commit()
        return status, True

    def unfollow(self, db: Session, follower_id: str, following_id: str) -> tuple[bool, str | None]:
        follow = db.scalar(
            select(Follow).where(Follow.follower_id == follower_id, Follow.following_id == following_id)
        )
        if follow:
            db.delete(follow)
            db.commit()
        return True, None

    def accept_follow(self, db: Session, athlete_id: str, follower_id: str) -> tuple[bool, str | None]:
        follow = db.scalar(
            select(Follow).where(
                Follow.follower_id == follower_id,
                Follow.following_id == athlete_id,
                Follow.status == "pending",
            )
        )
        if not follow:
            return False, "No pending request"
        follow.status = "following"
        db.commit()
        return True, None

    def deny_follow(self, db: Session, athlete_id: str, follower_id: str) -> tuple[bool, str | None]:
        follow = db.scalar(
            select(Follow).where(
                Follow.follower_id == follower_id,
                Follow.following_id == athlete_id,
                Follow.status == "pending",
            )
        )
        if follow:
            db.delete(follow)
            db.commit()
        return True, None

    def list_followers(self, db: Session, athlete_id: str):
        return (
            select(Athlete)
            .join(Follow, Follow.follower_id == Athlete.id)
            .where(Follow.following_id == athlete_id, Follow.status == "following")
        )

    def list_following(self, db: Session, athlete_id: str):
        return (
            select(Athlete)
            .join(Follow, Follow.following_id == Athlete.id)
            .where(Follow.follower_id == athlete_id, Follow.status == "following")
        )

    def list_follow_requests(self, db: Session, athlete_id: str):
        return (
            select(Athlete)
            .join(Follow, Follow.follower_id == Athlete.id)
            .where(Follow.following_id == athlete_id, Follow.status == "pending")
        )

    def add_comment(self, db: Session, author_id: str, target_type: str, target_id: str, text: str) -> CommentSchema:
        comment = Comment(author_id=author_id, target_type=target_type, target_id=target_id, text=text)
        db.add(comment)
        db.flush()
        if target_type == "activity":
            from app.models.activity import Activity
            activity = db.get(Activity, target_id)
            if activity:
                activity.comment_count += 1
                notification_service.create(
                    db,
                    activity.athlete_id,
                    "activity_comment",
                    author_id,
                    {"activity_id": target_id, "comment_id": comment.id},
                )
        elif target_type == "post":
            post = db.get(AthletePost, target_id)
            if post:
                post.comment_count += 1
                notification_service.create(
                    db,
                    post.athlete_id,
                    "post_comment",
                    author_id,
                    {"post_id": target_id, "comment_id": comment.id},
                )
        elif target_type == "club_post":
            club_post = db.get(ClubPost, target_id)
            if club_post:
                club_post.comment_count += 1
        author = db.get(Athlete, author_id)
        db.commit()
        return CommentSchema(
            id=comment.id,
            author=to_summary(author),
            text=comment.text,
            created_at=comment.created_at.isoformat(),
            like_count=0,
            is_liked=False,
        )

    def _content_owner_id(self, db: Session, target_type: str, target_id: str) -> str | None:
        if target_type == "activity":
            from app.models.activity import Activity
            activity = db.get(Activity, target_id)
            return activity.athlete_id if activity else None
        if target_type == "post":
            post = db.get(AthletePost, target_id)
            return post.athlete_id if post else None
        if target_type == "club_post":
            club_post = db.get(ClubPost, target_id)
            return club_post.author_id if club_post else None
        if target_type == "comment":
            comment = db.get(Comment, target_id)
            return comment.author_id if comment else None
        return None

    def like(self, db: Session, athlete_id: str, target_type: str, target_id: str) -> tuple[bool, str | None]:
        owner_id = self._content_owner_id(db, target_type, target_id)
        if owner_id is None:
            return False, "Content not found"
        if owner_id == athlete_id:
            return False, "You cannot like your own content"
        existing = db.scalar(
            select(Like).where(
                Like.athlete_id == athlete_id,
                Like.target_type == target_type,
                Like.target_id == target_id,
            )
        )
        if existing:
            return True, None
        db.add(Like(athlete_id=athlete_id, target_type=target_type, target_id=target_id))
        if target_type == "activity":
            from app.models.activity import Activity
            activity = db.get(Activity, target_id)
            if activity:
                activity.like_count += 1
                notification_service.create(
                    db, activity.athlete_id, "activity_like", athlete_id, {"activity_id": target_id}
                )
        elif target_type == "post":
            post = db.get(AthletePost, target_id)
            if post:
                post.like_count += 1
                notification_service.create(
                    db, post.athlete_id, "post_like", athlete_id, {"post_id": target_id}
                )
        elif target_type == "club_post":
            club_post = db.get(ClubPost, target_id)
            if club_post:
                club_post.like_count += 1
        elif target_type == "comment":
            comment = db.get(Comment, target_id)
            if comment:
                comment.like_count += 1
        db.commit()
        return True, None

    def unlike(self, db: Session, athlete_id: str, target_type: str, target_id: str) -> tuple[bool, str | None]:
        like = db.scalar(
            select(Like).where(
                Like.athlete_id == athlete_id,
                Like.target_type == target_type,
                Like.target_id == target_id,
            )
        )
        if not like:
            return True, None
        if target_type == "activity":
            from app.models.activity import Activity
            activity = db.get(Activity, target_id)
            if activity and activity.like_count > 0:
                activity.like_count -= 1
        elif target_type == "post":
            post = db.get(AthletePost, target_id)
            if post and post.like_count > 0:
                post.like_count -= 1
        elif target_type == "club_post":
            club_post = db.get(ClubPost, target_id)
            if club_post and club_post.like_count > 0:
                club_post.like_count -= 1
        elif target_type == "comment":
            comment = db.get(Comment, target_id)
            if comment and comment.like_count > 0:
                comment.like_count -= 1
        db.delete(like)
        db.commit()
        return True, None

    def comment_like_counts(self, db: Session, comment_ids: list[str]) -> dict[str, int]:
        if not comment_ids:
            return {}
        rows = db.execute(
            select(Like.target_id, func.count())
            .where(Like.target_type == "comment", Like.target_id.in_(comment_ids))
            .group_by(Like.target_id)
        ).all()
        return {target_id: count for target_id, count in rows}

    def list_likers(self, db: Session, target_type: str, target_id: str):
        return (
            select(Athlete)
            .join(Like, Like.athlete_id == Athlete.id)
            .where(Like.target_type == target_type, Like.target_id == target_id)
            .order_by(Like.created_at.desc())
        )

    def list_comments(self, db: Session, target_type: str, target_id: str):
        return (
            select(Comment)
            .where(Comment.target_type == target_type, Comment.target_id == target_id)
            .order_by(Comment.created_at.desc())
        )

    def create_post(self, db: Session, athlete_id: str, text: str, media_urls: list[str] | None) -> AthletePostSchema:
        post = AthletePost(athlete_id=athlete_id, text=text, media_urls=media_urls or [])
        db.add(post)
        db.commit()
        db.refresh(post)
        return AthletePostSchema(
            id=post.id,
            athlete_id=post.athlete_id,
            text=post.text,
            media_urls=post.media_urls,
            created_at=post.created_at.isoformat(),
            like_count=0,
            comment_count=0,
            is_liked=False,
        )

    def block(self, db: Session, blocker_id: str, blocked_id: str) -> tuple[bool, str | None]:
        if blocker_id == blocked_id:
            return False, "Cannot block yourself"
        if not db.scalar(select(Block).where(Block.blocker_id == blocker_id, Block.blocked_id == blocked_id)):
            db.add(Block(blocker_id=blocker_id, blocked_id=blocked_id))
            db.commit()
        return True, None

    def unblock(self, db: Session, blocker_id: str, blocked_id: str) -> tuple[bool, str | None]:
        block = db.scalar(select(Block).where(Block.blocker_id == blocker_id, Block.blocked_id == blocked_id))
        if block:
            db.delete(block)
            db.commit()
        return True, None

    def list_blocks(self, db: Session, blocker_id: str):
        return (
            select(Athlete)
            .join(Block, Block.blocked_id == Athlete.id)
            .where(Block.blocker_id == blocker_id)
        )


social_service = SocialService()
