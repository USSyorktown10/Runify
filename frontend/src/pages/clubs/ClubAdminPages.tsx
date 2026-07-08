import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { EmptyState } from "@/components/EmptyState";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export function ClubJoinRequestsPage() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["club-join-requests", id],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/clubs/${id}/join-requests?per_page=50`),
    enabled: !!id,
  });

  const accept = useMutation({
    mutationFn: (athleteId: string) => api.post(`/clubs/${id}/join-requests/${athleteId}/accept`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["club-join-requests", id] });
      qc.invalidateQueries({ queryKey: ["club", id] });
    },
  });

  const deny = useMutation({
    mutationFn: (athleteId: string) => api.post(`/clubs/${id}/join-requests/${athleteId}/deny`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["club-join-requests", id] }),
  });

  return (
    <section>
      <h1 className="title mb-8">Join requests</h1>
      <Link className="cactus-link text-sm mb-4 inline-block" to={`/clubs/${id}/settings`}>
        ← Back to settings
      </Link>
      {isLoading && <p className="text-muted">Loading…</p>}
      {!isLoading && data?.items.length === 0 && (
        <EmptyState title="No pending requests" description="Athletes who request to join will appear here." />
      )}
      <ul className="space-y-4 mt-4">
        {data?.items.map((a) => (
          <li key={a.id} className="flex items-center justify-between gap-4 card py-3">
            <div className="flex items-center gap-2">
              <AthleteAvatar athlete={a} size="sm" />
              <Link className="cactus-link" to={`/athletes/${a.id}`}>
                {athleteName(a)}
              </Link>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                className="btn-primary text-xs"
                onClick={() => accept.mutate(a.id)}
                disabled={accept.isPending}
              >
                Accept
              </button>
              <button
                type="button"
                className="btn-secondary text-xs"
                onClick={() => deny.mutate(a.id)}
                disabled={deny.isPending}
              >
                Deny
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
