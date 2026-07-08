import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { useFormatters } from "@/components/MetricGrid";
import { athleteName } from "@/lib/format";
import type { PaginatedClubLeaderboardResponse } from "@/types/api";

type Period = "this_week" | "last_week";
type Metric = "distance" | "activity_count" | "longest_distance" | "avg_pace" | "elevation_gain";

const METRICS: { id: Metric; label: string }[] = [
  { id: "distance", label: "Distance" },
  { id: "activity_count", label: "Runs" },
  { id: "longest_distance", label: "Longest" },
  { id: "avg_pace", label: "Avg pace" },
  { id: "elevation_gain", label: "Elev gain" },
];

export function ClubLeaderboardTab({ clubId }: { clubId: string }) {
  const [page, setPage] = useState(1);
  const [period, setPeriod] = useState<Period>("this_week");
  const [metric, setMetric] = useState<Metric>("distance");
  const { distance, pace } = useFormatters();

  const { data, isLoading } = useQuery({
    queryKey: ["club-leaderboard", clubId, page, period, metric],
    queryFn: () =>
      api.get<PaginatedClubLeaderboardResponse>(
        `/clubs/${clubId}/leaderboard?period=${period}&metric=${metric}&page=${page}&per_page=20`,
      ),
    enabled: !!clubId,
  });

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="flex gap-1">
          <button
            type="button"
            className={`text-xs font-semibold uppercase px-3 py-1.5 border ${
              period === "this_week" ? "border-accent text-accent" : "border-border text-muted"
            }`}
            onClick={() => {
              setPeriod("this_week");
              setPage(1);
            }}
          >
            This week
          </button>
          <button
            type="button"
            className={`text-xs font-semibold uppercase px-3 py-1.5 border ${
              period === "last_week" ? "border-accent text-accent" : "border-border text-muted"
            }`}
            onClick={() => {
              setPeriod("last_week");
              setPage(1);
            }}
          >
            Last week
          </button>
        </div>
        <select
          className="field-input text-xs py-1.5 w-auto"
          value={metric}
          onChange={(e) => {
            setMetric(e.target.value as Metric);
            setPage(1);
          }}
        >
          {METRICS.map((m) => (
            <option key={m.id} value={m.id}>
              Sort by {m.label}
            </option>
          ))}
        </select>
      </div>

      {data?.viewer_summary && (
        <div className="card mb-4 text-sm">
          <p className="text-muted text-xs uppercase font-semibold mb-1">Your stats</p>
          <div className="flex flex-wrap gap-4">
            <span>
              Rank: <strong>{data.viewer_summary.rank ?? "—"}</strong>
            </span>
            <span>Distance: {distance(data.viewer_summary.distance)}</span>
            <span>Runs: {data.viewer_summary.activity_count}</span>
            {data.viewer_summary.avg_pace != null && (
              <span>Pace: {pace(data.viewer_summary.avg_pace)}</span>
            )}
          </div>
        </div>
      )}

      {isLoading && <p className="text-muted">Loading leaderboard…</p>}
      <div className="overflow-x-auto">
        <table className="data-table mt-2">
          <thead>
            <tr>
              <th>#</th>
              <th>Athlete</th>
              <th>Distance</th>
              <th>Runs</th>
              <th>Longest</th>
              <th>Pace</th>
              <th>Elev</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((e) => (
              <tr key={e.athlete_id}>
                <td className="text-highlight font-semibold">{e.rank}</td>
                <td>
                  <Link className="flex items-center gap-2 hover:text-accent" to={`/athletes/${e.athlete_id}`}>
                    <AthleteAvatar athlete={e.athlete} size="sm" />
                    <span className="cactus-link">{athleteName(e.athlete)}</span>
                  </Link>
                </td>
                <td>{distance(e.distance)}</td>
                <td>{e.activity_count}</td>
                <td>
                  {e.longest_activity_id ? (
                    <Link className="cactus-link" to={`/activities/${e.longest_activity_id}`}>
                      {distance(e.longest_distance)}
                    </Link>
                  ) : (
                    "—"
                  )}
                </td>
                <td>{e.avg_pace != null ? pace(e.avg_pace) : "—"}</td>
                <td>{e.elevation_gain > 0 ? `${e.elevation_gain.toFixed(0)} m` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data && data.pagination.total_pages > 1 && (
        <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />
      )}
    </div>
  );
}
