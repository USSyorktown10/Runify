import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ActivityCard } from "@/components/FeedCard";
import { Pagination } from "@/components/Pagination";
import { EmptyState } from "@/components/EmptyState";
import { useAuth } from "@/context/AuthContext";
import type { PaginatedActivitiesResponse } from "@/types/api";

export function ActivitiesPage() {
  const [page, setPage] = useState(1);
  const { user } = useAuth();

  const { data, isLoading } = useQuery({
    queryKey: ["activities", page],
    queryFn: () => api.get<PaginatedActivitiesResponse>(`/activities?page=${page}&per_page=20`),
  });

  return (
    <section>
      <div className="page-header">
        <div>
          <h1 className="title text-3xl">Activities</h1>
          <p className="prose-runify text-muted mt-1">Your recorded workouts.</p>
        </div>
        <div className="flex gap-2">
          <Link className="btn-secondary text-sm" to="/activities/upload">
            Upload
          </Link>
          <Link className="btn-primary text-sm" to="/activities/new">
            New
          </Link>
        </div>
      </div>
      {isLoading && <p className="text-muted">Loading…</p>}
      {!isLoading && data?.items.length === 0 && (
        <EmptyState
          title="No activities yet"
          description="Upload a GPX file or create a manual entry."
          action={
            <Link className="btn-primary" to="/activities/upload">
              Upload activity
            </Link>
          }
        />
      )}
      <ul className="flex flex-col gap-4" role="list">
        {user &&
          data?.items.map((activity) => (
            <li key={activity.id} className="list-none">
              <ActivityCard activity={activity} athlete={user} createdAt={activity.start_date} />
            </li>
          ))}
      </ul>
      {data && <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />}
    </section>
  );
}
