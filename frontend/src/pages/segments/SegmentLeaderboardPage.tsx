import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { useFormatters } from "@/components/MetricGrid";
import { athleteFromLeaderboard } from "@/lib/avatar";
import { athleteName } from "@/lib/format";
import type { PaginatedLeaderboardResponse } from "@/types/api";

export function SegmentLeaderboardPage() {
  const { id } = useParams<{ id: string }>();
  const [page, setPage] = useState(1);
  const { duration } = useFormatters();

  const { data, isLoading } = useQuery({
    queryKey: ["segment-leaderboard", id, page],
    queryFn: () => api.get<PaginatedLeaderboardResponse>(`/segments/${id}/leaderboard?page=${page}&per_page=20`),
    enabled: !!id,
  });

  return (
    <section>
      <h1 className="title mb-4">Leaderboard</h1>
      <Link className="cactus-link text-xs mb-2 inline-block" to={`/segments/${id}`}>
        ← Back to segment
      </Link>
      {isLoading && <p className="text-muted">Loading…</p>}
      <table className="data-table mt-2">
        <thead>
          <tr>
            <th>#</th>
            <th>Athlete</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {data?.items.map((e) => {
            const athlete = athleteFromLeaderboard(e);
            return (
              <tr key={e.athlete_id}>
                <td className="text-highlight font-semibold">{e.rank}</td>
                <td>
                  <Link className="flex items-center gap-2 hover:text-accent" to={`/athletes/${e.athlete_id}`}>
                    <AthleteAvatar athlete={athlete} size="sm" />
                    <span className="cactus-link">{athleteName(athlete)}</span>
                  </Link>
                </td>
                <td>{duration(e.elapsed_time)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
