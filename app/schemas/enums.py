from enum import StrEnum


class MeasurementSystem(StrEnum):
    METRIC = "metric"
    IMPERIAL = "imperial"


class VisibilityLevel(StrEnum):
    PUBLIC = "public"
    FOLLOWERS = "followers"
    PRIVATE = "private"


class StreamResolution(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FilterSortOrder(StrEnum):
    NEWEST = "newest"
    OLDEST = "oldest"


class SocialRelationshipStatus(StrEnum):
    FOLLOWING = "following"
    PENDING = "pending"
    NONE = "none"
    BLOCKED = "blocked"


class ActivityType(StrEnum):
    RUN = "run"
    TRAIL_RUN = "trail_run"
    TREADMILL_RUN = "treadmill_run"
    TRACK_RUN = "track_run"


class FeedItemType(StrEnum):
    ACTIVITY = "activity"
    POST = "post"
    CLUB_POST = "club_post"


class ClubMemberRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
