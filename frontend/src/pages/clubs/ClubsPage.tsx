import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ClubAvatar } from "@/components/ClubAvatar";
import { Pagination } from "@/components/Pagination";
import type { PaginatedClubsResponse } from "@/types/api";

export function ClubsPage() {
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["clubs", page, query],
    queryFn: () =>
      api.get<PaginatedClubsResponse>(
        `/clubs?page=${page}&per_page=20${query ? `&query=${encodeURIComponent(query)}` : ""}`,
      ),
  });

  return (
    <section>
      <div className="flex justify-between items-end mb-8">
        <h1 className="title">Clubs</h1>
        <Link className="btn-primary text-sm" to="/clubs/new">
          Create club
        </Link>
      </div>
      <input
        className="field-input mb-6 max-w-md"
        placeholder="Search clubs…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setPage(1);
        }}
      />
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4">
        {data?.items.map((c) => (
          <li key={c.id} className="card">
            <Link className="flex items-center gap-3 hover:text-accent" to={`/clubs/${c.id}`}>
              <ClubAvatar club={c} size="md" />
              <div className="min-w-0">
                <span className="title text-base block truncate">{c.name}</span>
                <p className="text-muted text-xs mt-1">
                  {c.member_count} members {c.is_private && "· Private"}
                </p>
              </div>
            </Link>
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
