import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { NotificationItem } from "@/components/NotificationItem";
import { Pagination } from "@/components/Pagination";
import { BackButton } from "@/components/BackButton";
import type { Notification, PaginatedNotificationsResponse } from "@/types/api";

export function NotificationsPage() {
  const [page, setPage] = useState(1);
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["notifications", page],
    queryFn: () => api.get<PaginatedNotificationsResponse>(`/athlete/notifications?page=${page}&per_page=20`),
  });

  const markAll = useMutation({
    mutationFn: () => api.post("/athlete/notifications/read-all"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notifications"] });
      qc.invalidateQueries({ queryKey: ["notifications-count"] });
    },
  });

  const markRead = useMutation({
    mutationFn: (ids: string[]) => api.post("/athlete/notifications/read", { notification_ids: ids }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notifications"] });
      qc.invalidateQueries({ queryKey: ["notifications-count"] });
    },
  });

  const handleNavigate = (notification: Notification) => {
    if (!notification.is_read) {
      markRead.mutate([notification.id]);
    }
  };

  return (
    <section>
      <BackButton label="Back to Feed" />
      <div className="mb-8 flex items-end justify-between">
        <h1 className="title">Notifications</h1>
        {(data?.items.some((n) => !n.is_read) ?? false) && (
          <button type="button" className="btn-secondary text-sm" onClick={() => markAll.mutate()}>
            Mark all read
          </button>
        )}
      </div>
      {isLoading && <p className="text-muted">Loading…</p>}
      {!isLoading && data?.items.length === 0 && (
        <p className="text-muted">No notifications yet. Likes, comments, and follows will show up here.</p>
      )}
      <ul className="space-y-3">
        {data?.items.map((n) => (
          <li key={n.id}>
            <NotificationItem notification={n} onNavigate={handleNavigate} />
          </li>
        ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
