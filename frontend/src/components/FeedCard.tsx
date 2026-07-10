import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { MetricGrid } from "@/components/MetricGrid";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ClubAvatar } from "@/components/ClubAvatar";
import { MapView } from "@/components/MapView";
import { CommentList } from "@/components/CommentList";
import { ViewLikersTrigger } from "@/components/LikersPanel";
import { useFormatters } from "@/components/MetricGrid";
import { useAuth } from "@/context/AuthContext";
import { athleteName, formatDateTime } from "@/lib/format";
import type { AthletePost, ClubPost, FeedItem, SummaryActivity } from "@/types/api";

function ActivityIcon({ type, className = "w-5 h-5 text-accent" }: { type: string; className?: string }) {
  const t = type.toLowerCase();
  if (t === "run") {
    return (
      <svg fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className}>
        <title>Run</title>
        <path d="M8.688 0C8.025 0 7.38.215 6.85.613l-3.32 2.49-2.845.948A1 1 0 000 5c0 1.579.197 2.772.567 3.734.376.978.907 1.654 1.476 2.223.305.305.6.567.886.82.785.697 1.5 1.33 2.159 2.634 1.032 2.57 2.37 4.748 4.446 6.27C11.629 22.218 14.356 23 18 23c2.128 0 3.587-.553 4.549-1.411a4.378 4.378 0 001.408-2.628c.152-.987-.389-1.787-.967-2.25l-3.892-3.114a1 1 0 01-.329-.477l-3.094-9.726A2 2 0 0013.769 2h-1.436a2 2 0 00-1.2.4l-.57.428-.516-1.803A1.413 1.413 0 008.688 0zM8.05 2.213c.069-.051.143-.094.221-.127l1.168 4.086L12.333 4h1.436l.954 3H12v2h3.36l.318 1H13v2h3.314l.55 1.726a3 3 0 00.984 1.433l3.106 2.485c-.77.19-1.778.356-2.954.356-1.97 0-3.178-.431-4.046-1.087-.895-.677-1.546-1.675-2.251-3.056-.224-.437-.45-.907-.688-1.403C9.875 10.08 8.444 7.1 5.531 4.102zM3.743 5.14c2.902 2.858 4.254 5.664 5.441 8.126.25.517.49 1.018.738 1.502.732 1.432 1.55 2.777 2.827 3.74C14.053 19.495 15.72 20 18 20c1.492 0 2.754-.23 3.684-.479a2.285 2.285 0 01-.467.575c-.5.446-1.435.904-3.217.904-3.356 0-5.629-.718-7.284-1.931-1.663-1.22-2.823-3.028-3.788-5.44a1.012 1.012 0 00-.034-.076c-.853-1.708-1.947-2.673-2.79-3.417a14.61 14.61 0 01-.647-.593c-.431-.431-.775-.88-1.024-1.527-.21-.545-.367-1.271-.417-2.3z" />
      </svg>
    );
  }
  if (t === "ride" || t === "cycling" || t === "ebikeride" || t === "virtualride") {
    return (
      <svg fill="currentColor" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" className={className}>
        <title>Ride</title>
        <path d="M49.6 37.2c-4 0-7.2 3.2-7.2 7.2s3.2 7.2 7.2 7.2 7.2-3.2 7.2-7.2-3.2-7.2-7.2-7.2zm0 11.2c-2.2 0-4-1.8-4-4s1.8-4 4-4 4 1.8 4 4-1.8 4-4 4zM14.4 37.2c-4 0-7.2 3.2-7.2 7.2s3.2 7.2 7.2 7.2 7.2-3.2 7.2-7.2-3.2-7.2-7.2-7.2zm0 11.2c-2.2 0-4-1.8-4-4s1.8-4 4-4 4 1.8 4 4-1.8 4-4 4zm24.6-21.5c.9 1.4 2.4 2.3 4.2 2.3 2.8 0 5-2.2 5-5s-2.2-5-5-5-5 2.2-5 5c0 .3 0 .7.1 1L29.7 28.5c-.8-.9-1.9-1.5-3.2-1.5h-8.2V31h8.2c1 0 1.8.8 1.8 1.8v8.6c0 1.2.6 2.3 1.6 3l10 7.2 2.3-3.2-9-6.5V34l12-14.8z" />
      </svg>
    );
  }
  if (t === "swim") {
    return (
      <svg fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className}>
        <title>Swim</title>
        <path d="M2 6c3 0 3 2 6 2s3-2 6-2 3 2 6 2" />
        <path d="M2 12c3 0 3 2 6 2s3-2 6-2 3 2 6 2" />
        <path d="M2 18c3 0 3 2 6 2s3-2 6-2 3 2 6 2" />
      </svg>
    );
  }
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className}>
      <title>Activity</title>
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
    </svg>
  );
}

function formatFeedDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  }
  return `${m}m ${s}s`;
}

export function CardHeader({
  athlete,
  createdAt,
  deviceName,
}: {
  athlete: any;
  createdAt: string;
  deviceName?: string;
}) {
  const location = [athlete.city, athlete.state, athlete.country].filter(Boolean).join(", ");

  return (
    <div className="flex items-center gap-3 border-b border-border/40 pb-2 mb-3">
      <AthleteAvatar athlete={athlete} size="md" />
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-sm">
          <Link className="hover:text-accent transition-colors" to={`/athletes/${athlete.id}`}>
            {athleteName(athlete)}
          </Link>
        </div>
        <div className="text-muted text-[11px] flex flex-wrap items-center gap-x-1.5 gap-y-0.5 mt-0.5">
          <time>{formatDateTime(createdAt)}</time>
          {deviceName && (
            <>
              <span className="text-border/60">•</span>
              <span>{deviceName}</span>
            </>
          )}
          {location && (
            <>
              <span className="text-border/60">•</span>
              <span>{location}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function ActivityCard({
  activity,
  athlete,
  createdAt,
}: {
  activity: SummaryActivity;
  athlete?: any;
  createdAt?: string;
}) {
  const { user } = useAuth();
  const { distance, pace, system } = useFormatters();
  const queryClient = useQueryClient();
  const [showComments, setShowComments] = useState(false);
  const isOwner = user?.id === activity.athlete_id;

  const likeMutation = useMutation({
    mutationFn: () => {
      return activity.is_liked
        ? api.delete(`/activities/${activity.id}/likes`)
        : api.post(`/activities/${activity.id}/likes`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["activity", activity.id] });
      queryClient.invalidateQueries({ queryKey: ["likers", "activity", activity.id] });
    },
  });

  const avgSpeed = activity.moving_time > 0 ? activity.distance / activity.moving_time : 0;
  const isCycling = ["ride", "cycling", "ebikeride", "virtualride"].includes(activity.activity_type.toLowerCase());
  const paceLabel = isCycling ? "Speed" : "Pace";

  const formattedPace = isCycling
    ? system === "imperial"
      ? `${(avgSpeed * 2.23694).toFixed(1)} mph`
      : `${(avgSpeed * 3.6).toFixed(1)} km/h`
    : pace(avgSpeed);

  return (
    <article className="card flex flex-col justify-between">
      {athlete && createdAt && (
        <CardHeader athlete={athlete} createdAt={createdAt} deviceName={activity.device_name} />
      )}

      <div className="flex items-start gap-3">
        <div className="p-1.5 border border-border bg-surface shrink-0 flex items-center justify-center">
          <ActivityIcon type={activity.activity_type} className="w-5 h-5 text-accent" />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="subheading leading-snug">
            <Link className="hover:text-accent transition-colors" to={`/activities/${activity.id}`}>
              {activity.name}
            </Link>
          </h3>

          <div className="grid grid-cols-3 gap-2 border-t border-b border-border/40 py-2 mt-2 text-xs">
            <div>
              <span className="text-muted block text-[10px]  tracking-wider font-semibold">Distance</span>
              <span className="font-semibold text-sm tabular-nums">{distance(activity.distance)}</span>
            </div>
            <div>
              <span className="text-muted block text-[10px]  tracking-wider font-semibold">{paceLabel}</span>
              <span className="font-semibold text-sm tabular-nums">{formattedPace}</span>
            </div>
            <div>
              <span className="text-muted block text-[10px]  tracking-wider font-semibold font-mono">Time</span>
              <span className="font-semibold text-sm tabular-nums">{formatFeedDuration(activity.moving_time)}</span>
            </div>
          </div>

          {activity.metrics && activity.metrics.length > 0 && (
            <div className="mt-2">
              <MetricGrid metrics={activity.metrics.slice(0, 4)} />
            </div>
          )}
        </div>
      </div>

      {activity.polyline_summary && (
        <div className="mt-3 relative border border-border overflow-hidden group">
          <Link to={`/activities/${activity.id}`} className="block h-48 w-full hover:opacity-95 transition-opacity">
            <MapView polyline={activity.polyline_summary} className="h-full w-full pointer-events-none" interactive={false} />
            <div className="absolute top-2 right-2 bg-surface/90 border border-border px-2 py-0.5 text-[10px] font-semibold text-muted tracking-wider  backdrop-blur-sm shadow-sm">
              View Route
            </div>
          </Link>
        </div>
      )}

      <div className="flex items-center justify-between border-t border-border/40 pt-2 mt-3 text-xs">
        <div className="flex items-center gap-4">
          {!isOwner && (
            <button
              onClick={() => likeMutation.mutate()}
              disabled={likeMutation.isPending}
              className={`inline-flex items-center gap-1.5 font-semibold transition-colors disabled:opacity-50 py-1 hover:text-accent cursor-pointer ${
                activity.is_liked ? "text-accent" : "text-muted"
              }`}
              title={activity.is_liked ? "Unlike activity" : "Like this activity"}
            >
              <svg
                fill={activity.is_liked ? "currentColor" : "none"}
                stroke="currentColor"
                strokeWidth="1.5"
                viewBox="0 0 16 16"
                xmlns="http://www.w3.org/2000/svg"
                className="w-4 h-4"
              >
                <path d="M6.18.36A.625.625 0 016.746 0h.366a2.625 2.625 0 012.609 2.918L9.374 6h3.69a2.185 2.185 0 011.68 3.584l-.119.142v1.291c0 .458-.16.902-.454 1.254l-.171.205v.399A2.125 2.125 0 0111.875 15H5.703c-.256 0-.507-.077-.72-.22l-1.157-.777a.042.042 0 00-.024-.007l-1.483.031A1.292 1.292 0 011 12.736V8.81c0-.38.168-.742.46-.988l2.032-1.711zm.964.89L4.566 6.765a.625.625 0 01-.163.213l-2.138 1.8a.042.042 0 00-.015.032v3.926c0 .023.02.042.043.041l1.483-.03c.266-.006.527.07.748.219l1.156.777a.042.042 0 00.023.007h6.172a.875.875 0 00.875-.875v-.851l.46-.553a.708.708 0 00.165-.454V9.274l.408-.49a.935.935 0 00-.718-1.534h-5.09l.504-4.471c.09-.805-.53-1.51-1.335-1.529z" />
              </svg>
            </button>
          )}
          <ViewLikersTrigger
            targetType="activity"
            targetId={activity.id}
            likeCount={activity.like_count}
            title="Activity likes"
            className="text-xs font-semibold text-muted"
            isOwner={isOwner}
          />
        </div>

        <button
          onClick={() => setShowComments(!showComments)}
          className={`inline-flex items-center gap-1.5 font-semibold transition-colors py-1 cursor-pointer hover:text-accent ${
            showComments ? "text-accent" : "text-muted"
          }`}
          title="Toggle comments"
        >
          <svg
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            viewBox="0 0 16 16"
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4"
          >
            <path d="M3 5.75h10V7H3zM3 8v1.25h7V8z" fill="currentColor" stroke="none" />
            <path d="M0 3.958C0 2.877.877 2 1.958 2h12.084C15.123 2 16 2.877 16 3.958v7.084A1.958 1.958 0 0114.042 13H7.759l-2.636 2.636A1.243 1.243 0 013 14.756V13H1.958A1.958 1.958 0 010 11.042zm1.958-.708a.708.708 0 00-.708.708v7.084c0 .39.317.708.708.708H4.25v2.991l2.991-2.991h6.8a.708.708 0 00.709-.708V3.958a.708.708 0 00-.708-.708z" />
          </svg>
          <span>
            {activity.comment_count === 0
              ? "Add Comment"
              : `${activity.comment_count} ${activity.comment_count === 1 ? "comment" : "comments"}`}
          </span>
        </button>
      </div>

      {showComments && (
        <CommentList targetType="activity" targetId={activity.id} variant="inline" />
      )}
    </article>
  );
}

export function PostCard({
  post,
  athlete,
  createdAt,
}: {
  post: AthletePost;
  athlete: any;
  createdAt: string;
}) {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [showComments, setShowComments] = useState(false);
  const isOwner = user?.id === post.athlete_id;

  const likeMutation = useMutation({
    mutationFn: () => {
      return post.is_liked
        ? api.delete(`/posts/${post.id}/likes`)
        : api.post(`/posts/${post.id}/likes`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["athlete-feed"] });
      queryClient.invalidateQueries({ queryKey: ["likers", "post", post.id] });
    },
  });

  return (
    <article className="card flex flex-col">
      <CardHeader athlete={athlete} createdAt={createdAt} />
      <div className="text-sm leading-relaxed whitespace-pre-wrap mt-1 px-1">
        {post.text}
      </div>
      <div className="flex items-center justify-between border-t border-border/40 pt-2 mt-3 text-xs">
        <div className="flex items-center gap-4">
          {!isOwner && (
            <button
              onClick={() => likeMutation.mutate()}
              disabled={likeMutation.isPending}
              className={`inline-flex items-center gap-1.5 font-semibold transition-colors disabled:opacity-50 py-1 hover:text-accent cursor-pointer ${
                post.is_liked ? "text-accent" : "text-muted"
              }`}
              title={post.is_liked ? "Unlike post" : "Like this post"}
            >
              <svg
                fill={post.is_liked ? "currentColor" : "none"}
                stroke="currentColor"
                strokeWidth="1.5"
                viewBox="0 0 16 16"
                xmlns="http://www.w3.org/2000/svg"
                className="w-4 h-4"
              >
                <path d="M6.18.36A.625.625 0 016.746 0h.366a2.625 2.625 0 012.609 2.918L9.374 6h3.69a2.185 2.185 0 011.68 3.584l-.119.142v1.291c0 .458-.16.902-.454 1.254l-.171.205v.399A2.125 2.125 0 0111.875 15H5.703c-.256 0-.507-.077-.72-.22l-1.157-.777a.042.042 0 00-.024-.007l-1.483.031A1.292 1.292 0 011 12.736V8.81c0-.38.168-.742.46-.988l2.032-1.711zm.964.89L4.566 6.765a.625.625 0 01-.163.213l-2.138 1.8a.042.042 0 00-.015.032v3.926c0 .023.02.042.043.041l1.483-.03c.266-.006.527.07.748.219l1.156.777a.042.042 0 00.023.007h6.172a.875.875 0 00.875-.875v-.851l.46-.553a.708.708 0 00.165-.454V9.274l.408-.49a.935.935 0 00-.718-1.534h-5.09l.504-4.471c.09-.805-.53-1.51-1.335-1.529z" />
              </svg>
            </button>
          )}
          <ViewLikersTrigger
            targetType="post"
            targetId={post.id}
            likeCount={post.like_count}
            title="Post likes"
            className="text-xs font-semibold text-muted"
            isOwner={isOwner}
          />
        </div>

        <button
          onClick={() => setShowComments(!showComments)}
          className={`inline-flex items-center gap-1.5 font-semibold transition-colors py-1 cursor-pointer hover:text-accent ${
            showComments ? "text-accent" : "text-muted"
          }`}
          title="Toggle comments"
        >
          <svg
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            viewBox="0 0 16 16"
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4"
          >
            <path d="M3 5.75h10V7H3zM3 8v1.25h7V8z" fill="currentColor" stroke="none" />
            <path d="M0 3.958C0 2.877.877 2 1.958 2h12.084C15.123 2 16 2.877 16 3.958v7.084A1.958 1.958 0 0114.042 13H7.759l-2.636 2.636A1.243 1.243 0 013 14.756V13H1.958A1.958 1.958 0 010 11.042zm1.958-.708a.708.708 0 00-.708.708v7.084c0 .39.317.708.708.708H4.25v2.991l2.991-2.991h6.8a.708.708 0 00.709-.708V3.958a.708.708 0 00-.708-.708z" />
          </svg>
          <span>
            {post.comment_count === 0
              ? "Add Comment"
              : `${post.comment_count} ${post.comment_count === 1 ? "comment" : "comments"}`}
          </span>
        </button>
      </div>

      {showComments && (
        <CommentList targetType="post" targetId={post.id} variant="inline" />
      )}
    </article>
  );
}

export function ClubPostCard({
  clubPost,
  athlete,
  createdAt,
}: {
  clubPost: ClubPost;
  athlete: any;
  createdAt: string;
}) {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [showComments, setShowComments] = useState(false);
  const isOwner = user?.id === clubPost.author.id;

  const likeMutation = useMutation({
    mutationFn: () => {
      return clubPost.is_liked
        ? api.delete(`/club-posts/${clubPost.id}/likes`)
        : api.post(`/club-posts/${clubPost.id}/likes`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["likers", "club_post", clubPost.id] });
    },
  });

  return (
    <article className="card flex flex-col">
      <CardHeader athlete={athlete} createdAt={createdAt} />
      <div className="mt-1 px-1">
        <Link
          to={`/clubs/${clubPost.club_id}`}
          className="inline-flex items-center gap-2 mb-2 text-xs text-muted hover:text-accent transition-colors"
        >
          <ClubAvatar club={clubPost.club} size="sm" />
          <span className="font-semibold">{clubPost.club.name}</span>
        </Link>
        <h3 className="title text-base">{clubPost.title}</h3>
        <p className="text-sm leading-relaxed whitespace-pre-wrap mt-2">{clubPost.body}</p>
      </div>
      <div className="flex items-center justify-between border-t border-border/40 pt-2 mt-3 text-xs">
        <div className="flex items-center gap-4">
          {!isOwner && (
            <button
              onClick={() => likeMutation.mutate()}
              disabled={likeMutation.isPending}
              className={`inline-flex items-center gap-1.5 font-semibold transition-colors disabled:opacity-50 py-1 hover:text-accent cursor-pointer ${
                clubPost.is_liked ? "text-accent" : "text-muted"
              }`}
              title={clubPost.is_liked ? "Unlike post" : "Like this post"}
            >
              <svg
                fill={clubPost.is_liked ? "currentColor" : "none"}
                stroke="currentColor"
                strokeWidth="1.5"
                viewBox="0 0 16 16"
                xmlns="http://www.w3.org/2000/svg"
                className="w-4 h-4"
              >
                <path d="M6.18.36A.625.625 0 016.746 0h.366a2.625 2.625 0 012.609 2.918L9.374 6h3.69a2.185 2.185 0 011.68 3.584l-.119.142v1.291c0 .458-.16.902-.454 1.254l-.171.205v.399A2.125 2.125 0 0111.875 15H5.703c-.256 0-.507-.077-.72-.22l-1.157-.777a.042.042 0 00-.024-.007l-1.483.031A1.292 1.292 0 011 12.736V8.81c0-.38.168-.742.46-.988l2.032-1.711zm.964.89L4.566 6.765a.625.625 0 01-.163.213l-2.138 1.8a.042.042 0 00-.015.032v3.926c0 .023.02.042.043.041l1.483-.03c.266-.006.527.07.748.219l1.156.777a.042.042 0 00.023.007h6.172a.875.875 0 00.875-.875v-.851l.46-.553a.708.708 0 00.165-.454V9.274l.408-.49a.935.935 0 00-.718-1.534h-5.09l.504-4.471c.09-.805-.53-1.51-1.335-1.529z" />
              </svg>
            </button>
          )}
          <ViewLikersTrigger
            targetType="club_post"
            targetId={clubPost.id}
            likeCount={clubPost.like_count}
            title="Post likes"
            className="text-xs font-semibold text-muted"
            isOwner={isOwner}
          />
        </div>

        <button
          onClick={() => setShowComments(!showComments)}
          className={`inline-flex items-center gap-1.5 font-semibold transition-colors py-1 cursor-pointer hover:text-accent ${
            showComments ? "text-accent" : "text-muted"
          }`}
          title="Toggle comments"
        >
          <svg
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            viewBox="0 0 16 16"
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4"
          >
            <path d="M3 5.75h10V7H3zM3 8v1.25h7V8z" fill="currentColor" stroke="none" />
            <path d="M0 3.958C0 2.877.877 2 1.958 2h12.084C15.123 2 16 2.877 16 3.958v7.084A1.958 1.958 0 0114.042 13H7.759l-2.636 2.636A1.243 1.243 0 013 14.756V13H1.958A1.958 1.958 0 010 11.042zm1.958-.708a.708.708 0 00-.708.708v7.084c0 .39.317.708.708.708H4.25v2.991l2.991-2.991h6.8a.708.708 0 00.709-.708V3.958a.708.708 0 00-.708-.708z" />
          </svg>
          <span>
            {clubPost.comment_count === 0
              ? "Add Comment"
              : `${clubPost.comment_count} ${clubPost.comment_count === 1 ? "comment" : "comments"}`}
          </span>
        </button>
      </div>

      {showComments && (
        <CommentList targetType="club_post" targetId={clubPost.id} variant="inline" />
      )}
    </article>
  );
}

export function FeedCard({ item }: { item: FeedItem }) {
  return (
    <li className="list-none">
      {item.type === "activity" && item.activity_data && (
        <ActivityCard activity={item.activity_data} athlete={item.athlete} createdAt={item.created_at} />
      )}
      {item.type === "post" && item.post_data && (
        <PostCard post={item.post_data} athlete={item.athlete} createdAt={item.created_at} />
      )}
      {item.type === "club_post" && item.club_post_data && (
        <ClubPostCard clubPost={item.club_post_data} athlete={item.athlete} createdAt={item.created_at} />
      )}
    </li>
  );
}
