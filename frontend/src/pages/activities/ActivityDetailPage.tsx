import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { CommentList } from "@/components/CommentList";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { MapView } from "@/components/MapView";
import { MetricGrid } from "@/components/MetricGrid";
import { ReportDialog } from "@/components/ReportDialog";
import { PrivacyBadge, PrivacyVeil } from "@/components/PrivacyBadge";
import { DistributionChart, StreamChart, ZoneChart } from "@/components/StreamChart";
import { useFormatters } from "@/components/MetricGrid";
import { useAuth } from "@/context/AuthContext";
import { canViewBiometrics } from "@/lib/privacy";
import { LikersPanel } from "@/components/LikersPanel";
import { BackButton } from "@/components/BackButton";
import { athleteName } from "@/lib/format";
import type { DetailedActivity, DetailedAthlete, PowerCurve, Split, Stream } from "@/types/api";


export function ActivityDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { distance, duration, date } = useFormatters();
  const [reportOpen, setReportOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const { data: activity, isLoading, error } = useQuery({
    queryKey: ["activity", id],
    queryFn: () => api.get<DetailedActivity>(`/activities/${id}`),
    enabled: !!id,
  });

  const { data: athlete } = useQuery({
    queryKey: ["athlete", activity?.athlete_id],
    queryFn: () => api.get<DetailedAthlete>(`/athletes/${activity!.athlete_id}`),
    enabled: !!activity?.athlete_id,
  });

  const { data: streams } = useQuery({
    queryKey: ["activity-streams", id],
    queryFn: () =>
      api.get<Stream[]>(
        `/activities/${id}/streams?streams=heart_rate,pace,altitude&resolution=medium`,
      ),
    enabled: !!id && !!activity,
  });

  const { data: splits } = useQuery({
    queryKey: ["activity-splits", id],
    queryFn: () => api.get<Split[]>(`/activities/${id}/splits`),
    enabled: !!id,
  });

  const { data: powerCurve } = useQuery({
    queryKey: ["activity-power", id],
    queryFn: () => api.get<PowerCurve>(`/activities/${id}/power-curve`),
    enabled: !!id,
  });

  const like = useMutation({
    mutationFn: () =>
      activity?.is_liked
        ? api.delete(`/activities/${id}/likes`)
        : api.post(`/activities/${id}/likes`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["activity", id] });
      qc.invalidateQueries({ queryKey: ["likers", "activity", id] });
    },
  });

  const remove = useMutation({
    mutationFn: () => api.delete(`/activities/${id}`),
    onSuccess: () => navigate("/activities"),
  });

  if (isLoading) return <p className="text-muted">Loading…</p>;
  if (error || !activity) return <p className="text-red-500">Activity not found.</p>;

  const isOwner = user?.id === activity.athlete_id;
  const showBio = canViewBiometrics(activity, isOwner, false);

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_17rem] items-start">
      <section>
      <BackButton label="Back to Feed" />
      {athlete && (
        <Link
          to={`/athletes/${athlete.id}`}
          className="mb-4 inline-flex items-center gap-2 text-sm hover:text-accent"
        >
          <AthleteAvatar athlete={athlete} size="sm" />
          <span className="font-semibold">{athleteName(athlete)}</span>
        </Link>
      )}
      <div className="page-header">
        <div>
          <time className="meta font-semibold">{date(activity.start_date)}</time>
          <h1 className="title mt-1 text-3xl sm:text-4xl">{activity.name}</h1>
          <PrivacyBadge visibility={activity.visibility} />
        </div>
        {isOwner && (
          <div className="flex flex-wrap gap-2">
            <Link className="btn-secondary text-sm" to={`/activities/${id}/edit`}>
              Edit
            </Link>
            <Link className="btn-secondary text-sm" to={`/activities/${id}/crop`}>
              Crop
            </Link>
            <button type="button" className="btn-secondary text-sm text-red-500" onClick={() => setDeleteOpen(true)}>
              Delete
            </button>
          </div>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <div>
          {activity.polyline && (
            <MapView polyline={activity.polyline} className="h-[28rem] w-full rounded-none border border-border" />
          )}
        </div>
        <div>
          <p className="prose-runify mb-2">
            {distance(activity.distance)} · Moving {duration(activity.moving_time)}
            {activity.device_name && ` · ${activity.device_name}`}
          </p>
          {activity.description && <p className="text-muted mb-3">{activity.description}</p>}

          {showBio ? (
            <MetricGrid metrics={activity.metrics} />
          ) : (
            <PrivacyVeil />
          )}
        </div>
      </div>

      {showBio && (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          {activity.zones.map((z) => (
            <ZoneChart key={z.key} zone={z} />
          ))}
          {activity.distributions.map((d) => (
            <DistributionChart key={d.key} dist={d} />
          ))}
          {streams?.map((s) => (
            <StreamChart key={s.metric_key} stream={s} />
          ))}
        </div>
      )}

      {splits && splits.length > 0 && (
        <section className="mt-4">
          <h2 className="title mb-2">Splits</h2>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Time</th>
                <th>Elev</th>
              </tr>
            </thead>
            <tbody>
              {splits.map((s) => (
                <tr key={s.index}>
                  <td>{s.index + 1}</td>
                  <td>{duration(s.elapsed_time)}</td>
                  <td>{s.elevation_difference.toFixed(0)} m</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {powerCurve && powerCurve.curve_values.length > 0 && showBio && (
        <section className="mt-4">
          <h2 className="title mb-2">Power curve</h2>
          <ul className="text-xs space-y-0.5 font-mono">
            {powerCurve.curve_values.slice(0, 10).map((p) => (
              <li key={p.time_interval_seconds}>
                {p.time_interval_seconds}s: <strong>{p.power_value_watts} W</strong>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="flex flex-wrap items-center gap-2 mt-4">
        {!isOwner && (
          <button type="button" className="btn-secondary" onClick={() => like.mutate()}>
            {activity.is_liked ? "Unlike" : "Like"}
          </button>
        )}
        <span className="text-sm text-muted">
          {activity.like_count === 0 && isOwner
            ? "No likes yet"
            : `${activity.like_count} ${activity.like_count === 1 ? "like" : "likes"}`}
        </span>
        <button type="button" className="btn-secondary" onClick={() => setReportOpen(true)}>
          Report
        </button>
      </div>

      <CommentList targetType="activity" targetId={activity.id} />

      <ReportDialog open={reportOpen} targetType="activity" targetId={activity.id} onClose={() => setReportOpen(false)} />
      <ConfirmDialog
        open={deleteOpen}
        title="Delete activity"
        message="This cannot be undone."
        confirmLabel="Delete"
        danger
        onConfirm={() => remove.mutate()}
        onCancel={() => setDeleteOpen(false)}
      />
      </section>

      <aside className="lg:sticky lg:top-20 space-y-4">
        <LikersPanel
          targetType="activity"
          targetId={activity.id}
          likeCount={activity.like_count}
          title="Liked by"
          isOwner={isOwner}
        />
      </aside>
    </div>
  );
}
