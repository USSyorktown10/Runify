import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ActivityCard } from "@/components/FeedCard";
import { Pagination } from "@/components/Pagination";
import { EmptyState } from "@/components/EmptyState";
import type { PaginatedActivitiesResponse, PaginatedAthletesResponse } from "@/types/api";

export function ClubRecentActivityTab({ clubId }: { clubId: string }) {
  const [page, setPage] = useState(1);

  const { data: members } = useQuery({
    queryKey: ["club-members-map", clubId],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/clubs/${clubId}/members?per_page=100`),
    enabled: !!clubId,
  });

  const memberMap = useMemo(
    () => new Map(members?.items.map((a) => [a.id, a]) ?? []),
    [members],
  );

  const { data, isLoading } = useQuery({
    queryKey: ["club-recent-activity", clubId, page],
    queryFn: () =>
      api.get<PaginatedActivitiesResponse>(`/clubs/${clubId}/recent-activity?page=${page}&per_page=10`),
    enabled: !!clubId,
  });

  return (
    <div>
      {isLoading && <p className="text-muted">Loading activities…</p>}
      {!isLoading && data?.items.length === 0 && (
        <EmptyState title="No recent activity" description="Member runs will show up here." />
      )}
      <ul className="space-y-4">
        {data?.items.map((activity) => {
          const athlete = memberMap.get(activity.athlete_id);
          if (!athlete) return null;
          return (
            <li key={activity.id} className="list-none">
              <ActivityCard activity={activity} athlete={athlete} createdAt={activity.start_date} />
            </li>
          );
        })}
      </ul>
      {data && data.pagination.total_pages > 1 && (
        <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />
      )}
    </div>
  );
}
