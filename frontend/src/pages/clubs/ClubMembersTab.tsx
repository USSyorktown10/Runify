import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export function ClubMembersTab({ clubId }: { clubId: string }) {
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["club-members", clubId, page, query],
    queryFn: () => {
      const params = new URLSearchParams({ page: String(page), per_page: "20" });
      if (query) params.set("query", query);
      return api.get<PaginatedAthletesResponse>(`/clubs/${clubId}/members?${params}`);
    },
    enabled: !!clubId,
  });

  return (
    <div>
      <input
        className="field-input mb-4 max-w-sm"
        placeholder="Search members…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setPage(1);
        }}
      />
      {isLoading && <p className="text-muted">Loading members…</p>}
      <ul className="space-y-3">
        {data?.items.map((a) => (
          <li key={a.id} className="flex items-center gap-2">
            <AthleteAvatar athlete={a} size="sm" />
            <Link className="cactus-link" to={`/athletes/${a.id}`}>
              {athleteName(a)}
            </Link>
          </li>
        ))}
      </ul>
      {data && data.pagination.total_pages > 1 && (
        <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />
      )}
    </div>
  );
}
