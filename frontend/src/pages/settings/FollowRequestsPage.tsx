import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { FollowRequestActions } from "@/components/RelationshipButton";
import { Pagination } from "@/components/Pagination";
import { BackButton } from "@/components/BackButton";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export function FollowRequestsPage() {
  const [page, setPage] = useState(1);

  const { data, isLoading } = useQuery({
    queryKey: ["follow-requests", page],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/athlete/follow-requests?page=${page}&per_page=20`),
  });

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Follow requests</h1>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4">
        {data?.items.map((a) => (
          <li key={a.id} className="card flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <AthleteAvatar athlete={a} size="sm" />
              <Link className="cactus-link" to={`/athletes/${a.id}`}>
                {athleteName(a)}
              </Link>
            </div>
            <FollowRequestActions athleteId={a.id} />
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
