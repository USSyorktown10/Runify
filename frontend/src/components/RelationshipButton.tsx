import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";

export function RelationshipButton({ athleteId }: { athleteId: string }) {
  const qc = useQueryClient();

  const { data: status } = useQuery({
    queryKey: ["relationship", athleteId],
    queryFn: () => api.get<{ status: string }>(`/athletes/${athleteId}/relationship`),
  });

  const follow = useMutation({
    mutationFn: () => api.post(`/athletes/${athleteId}/follow`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationship", athleteId] }),
  });

  const unfollow = useMutation({
    mutationFn: () => api.delete(`/athletes/${athleteId}/follow`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationship", athleteId] }),
  });

  const s = status?.status ?? "none";

  if (s === "following") {
    return (
      <button type="button" className="btn-secondary" onClick={() => unfollow.mutate()}>
        Unfollow
      </button>
    );
  }
  if (s === "pending") {
    return <span className="text-muted text-sm">Request pending</span>;
  }
  if (s === "blocked") {
    return <span className="text-muted text-sm">Blocked</span>;
  }

  return (
    <button type="button" className="btn-primary" onClick={() => follow.mutate()}>
      Follow
    </button>
  );
}

export function FollowRequestActions({ athleteId }: { athleteId: string }) {
  const qc = useQueryClient();
  const accept = useMutation({
    mutationFn: () => api.post(`/athletes/${athleteId}/follow/accept`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["follow-requests"] }),
  });
  const deny = useMutation({
    mutationFn: () => api.post(`/athletes/${athleteId}/follow/deny`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["follow-requests"] }),
  });

  return (
    <div className="flex gap-2">
      <button type="button" className="btn-primary text-xs" onClick={() => accept.mutate()}>
        Accept
      </button>
      <button type="button" className="btn-secondary text-xs" onClick={() => deny.mutate()}>
        Deny
      </button>
    </div>
  );
}
