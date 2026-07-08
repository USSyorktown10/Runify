import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export function ClubMembersPage() {
  const { id } = useParams<{ id: string }>();
  const [page, setPage] = useState(1);

  const { data, isLoading } = useQuery({
    queryKey: ["club-members", id, page],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/clubs/${id}/members?page=${page}&per_page=20`),
    enabled: !!id,
  });

  return (
    <section>
      <h1 className="title mb-8">Members</h1>
      <Link className="cactus-link text-sm mb-4 inline-block" to={`/clubs/${id}`}>
        ← Back to club
      </Link>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4 mt-4">
        {data?.items.map((a) => (
          <li key={a.id} className="flex items-center gap-2">
            <AthleteAvatar athlete={a} size="sm" />
            <Link className="cactus-link" to={`/athletes/${a.id}`}>
              {athleteName(a)}
            </Link>
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
