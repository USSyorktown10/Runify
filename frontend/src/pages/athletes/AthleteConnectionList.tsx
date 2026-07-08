import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export function AthleteConnectionList({
  athleteId,
  type,
}: {
  athleteId: string;
  type: "followers" | "following";
}) {
  const [page, setPage] = useState(1);

  const { data, isLoading } = useQuery({
    queryKey: [type, athleteId, page],
    queryFn: () =>
      api.get<PaginatedAthletesResponse>(`/athletes/${athleteId}/${type}?page=${page}&per_page=20`),
    enabled: !!athleteId,
  });

  if (isLoading) {
    return <p className="text-muted py-8">Loading…</p>;
  }

  if (!data?.items.length) {
    return (
      <p className="text-muted py-8">
        {type === "following" ? "Not following anyone yet." : "No followers yet."}
      </p>
    );
  }

  return (
    <div>
      <ul className="space-y-4">
        {data.items.map((a) => (
          <li key={a.id} className="flex items-center gap-3">
            <AthleteAvatar athlete={a} size="sm" />
            <Link className="cactus-link" to={`/athletes/${a.id}`}>
              {athleteName(a)}
            </Link>
          </li>
        ))}
      </ul>
      {data.pagination.total_pages > 1 && (
        <div className="mt-6">
          <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />
        </div>
      )}
    </div>
  );
}
