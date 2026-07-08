import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Pagination } from "@/components/Pagination";
import { useFormatters } from "@/components/MetricGrid";
import type { PaginatedSegmentsResponse } from "@/types/api";

export function SegmentsPage() {
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");
  const { distance } = useFormatters();

  const { data, isLoading } = useQuery({
    queryKey: ["segments", page, query],
    queryFn: () =>
      api.get<PaginatedSegmentsResponse>(
        `/segments?page=${page}&per_page=20${query ? `&query=${encodeURIComponent(query)}` : ""}`,
      ),
  });

  return (
    <section>
      <h1 className="title mb-8">Explore segments</h1>
      <input
        className="field-input mb-6 max-w-md"
        placeholder="Search segments…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setPage(1);
        }}
      />
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {data?.items.map((s) => (
          <li key={s.id} className="card">
            <Link className="cactus-link title text-base" to={`/segments/${s.id}`}>
              {s.name}
            </Link>
            <p className="text-muted text-xs mt-1">
              {distance(s.distance)} · Grade {s.average_grade.toFixed(1)}%
              {s.is_starred && " · ★"}
            </p>
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
