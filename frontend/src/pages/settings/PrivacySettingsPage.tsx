import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { BackButton } from "@/components/BackButton";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export function PrivacySettingsPage() {
  const [page, setPage] = useState(1);
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["blocks", page],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/athlete/blocks?page=${page}&per_page=20`),
  });

  const unblock = useMutation({
    mutationFn: (id: string) => api.delete(`/athletes/${id}/block`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["blocks"] }),
  });

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Blocked users</h1>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4">
        {data?.items.map((a) => (
          <li key={a.id} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AthleteAvatar athlete={a} size="sm" />
              <Link className="cactus-link" to={`/athletes/${a.id}`}>
                {athleteName(a)}
              </Link>
            </div>
            <button type="button" className="btn-secondary text-sm" onClick={() => unblock.mutate(a.id)}>
              Unblock
            </button>
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
