import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Pagination } from "@/components/Pagination";
import { useAuth } from "@/context/AuthContext";
import { useFormatters } from "@/components/MetricGrid";
import type { PaginatedRoutesResponse } from "@/types/api";

export function RoutesPage() {
  const { user } = useAuth();
  const [page, setPage] = useState(1);
  const { distance } = useFormatters();

  const { data, isLoading } = useQuery({
    queryKey: ["routes", user?.id, page],
    queryFn: () => api.get<PaginatedRoutesResponse>(`/athletes/${user!.id}/routes?page=${page}&per_page=20`),
    enabled: !!user,
  });

  return (
    <section>
      <div className="flex justify-between items-end mb-8">
        <h1 className="title">Routes</h1>
        <Link className="btn-primary text-sm" to="/routes/new">
          New route
        </Link>
      </div>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4">
        {data?.items.map((r) => (
          <li key={r.id}>
            <Link className="cactus-link" to={`/routes/${r.id}`}>
              {r.name}
            </Link>
            <span className="text-muted text-xs ms-2">{distance(r.distance)}</span>
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
