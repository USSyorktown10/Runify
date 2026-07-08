import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Pagination } from "@/components/Pagination";
import { useFormatters } from "@/components/MetricGrid";
import type { PaginatedSegmentEffortsResponse } from "@/types/api";

export function SegmentEffortsPage() {
  const { id } = useParams<{ id: string }>();
  const [page, setPage] = useState(1);
  const { duration, date } = useFormatters();

  const { data, isLoading } = useQuery({
    queryKey: ["segment-efforts", id, page],
    queryFn: () => api.get<PaginatedSegmentEffortsResponse>(`/segments/${id}/efforts?page=${page}&per_page=20`),
    enabled: !!id,
  });

  return (
    <section>
      <h1 className="title mb-8">Efforts</h1>
      <Link className="cactus-link text-sm mb-4 inline-block" to={`/segments/${id}`}>
        ← Back to segment
      </Link>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4 mt-4">
        {data?.items.map((e) => (
          <li key={e.id} className="grid gap-1 sm:grid-cols-[auto_1fr]">
            <time className="text-muted min-w-30 font-semibold">{date(e.start_date)}</time>
            <div>
              <Link className="cactus-link" to={`/activities/${e.activity_id}`}>
                {duration(e.elapsed_time)}
              </Link>
              {e.rank && <span className="text-highlight text-xs ms-2">#{e.rank}</span>}
            </div>
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
