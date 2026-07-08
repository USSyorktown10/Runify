import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { formatDateTime } from "@/lib/format";
import { BackButton } from "@/components/BackButton";
import type { ActiveSession } from "@/types/api";

export function SessionsSettingsPage() {
  const qc = useQueryClient();

  const { data: sessions, isLoading } = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.get<ActiveSession[]>("/authentication/sessions"),
  });

  const revoke = useMutation({
    mutationFn: (id: string) => api.delete(`/authentication/sessions/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });

  const revokeAll = useMutation({
    mutationFn: () => api.delete("/authentication/sessions?terminate_current=false"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Active sessions</h1>
      <button type="button" className="btn-secondary mb-6" onClick={() => revokeAll.mutate()}>
        Sign out other sessions
      </button>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4">
        {sessions?.map((s) => (
          <li key={s.session_id} className="card">
            <p className="font-semibold">
              {s.client_metadata.browser_name || "Browser"} {s.is_current && "(current)"}
            </p>
            <p className="text-muted text-xs">{s.ip_address} · {s.location}</p>
            <p className="text-muted text-xs">Last active {formatDateTime(s.last_active_at)}</p>
            {!s.is_current && (
              <button type="button" className="text-red-500 text-sm mt-2" onClick={() => revoke.mutate(s.session_id)}>
                Revoke
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
