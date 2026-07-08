import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Pagination } from "@/components/Pagination";
import { useFormatters } from "@/components/MetricGrid";
import type { PaginatedSegmentsResponse } from "@/types/api";

export function AthleteSegmentList({ athleteId }: { athleteId: string }) {
  const [page, setPage] = useState(1);
  const { distance } = useFormatters();

  const { data, isLoading } = useQuery({
    queryKey: ["athlete-segments", athleteId, page],
    queryFn: () =>
      api.get<PaginatedSegmentsResponse>(`/athletes/${athleteId}/segments?page=${page}&per_page=20`),
    enabled: !!athleteId,
  });

  if (isLoading) {
    return <p className="text-muted py-8">Loading…</p>;
  }

  if (!data?.items.length) {
    return <p className="text-muted py-8">No segments yet.</p>;
  }

  return (
    <div>
      <ul className="space-y-4">
        {data.items.map((s) => (
          <li key={s.id}>
            <Link className="cactus-link" to={`/segments/${s.id}`}>
              {s.name}
            </Link>
            <span className="text-muted text-xs ms-2">{distance(s.distance)}</span>
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
