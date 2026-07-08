import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { Pagination } from "@/components/Pagination";
import { athleteName } from "@/lib/format";
import type { PaginatedAthletesResponse } from "@/types/api";

export type LikeTargetType = "activity" | "post" | "club_post" | "comment";

function likersPath(targetType: LikeTargetType, targetId: string): string {
  switch (targetType) {
    case "activity":
      return `/activities/${targetId}/likes`;
    case "post":
      return `/posts/${targetId}/likes`;
    case "club_post":
      return `/club-posts/${targetId}/likes`;
    case "comment":
      return `/comments/${targetId}/likes`;
  }
}

export function LikersList({
  targetType,
  targetId,
  compact = false,
}: {
  targetType: LikeTargetType;
  targetId: string;
  compact?: boolean;
}) {
  const [page, setPage] = useState(1);

  const { data, isLoading } = useQuery({
    queryKey: ["likers", targetType, targetId, page],
    queryFn: () =>
      api.get<PaginatedAthletesResponse>(
        `${likersPath(targetType, targetId)}?page=${page}&per_page=${compact ? 10 : 20}`,
      ),
    enabled: !!targetId,
  });

  if (isLoading) {
    return <p className="text-muted text-xs">Loading…</p>;
  }

  if (!data?.items.length) {
    return <p className="text-muted text-xs">No likes yet.</p>;
  }

  return (
    <div>
      <ul className={compact ? "space-y-2" : "space-y-3"}>
        {data.items.map((a) => (
          <li key={a.id}>
            <Link className="flex items-center gap-2.5 hover:text-accent" to={`/athletes/${a.id}`}>
              <AthleteAvatar athlete={a} size="sm" />
              <div className="min-w-0">
                <span className="block truncate text-sm font-semibold">{athleteName(a)}</span>
                {(a.city || a.country) && (
                  <span className="block truncate text-[10px] text-muted">
                    {[a.city, a.country].filter(Boolean).join(", ")}
                  </span>
                )}
              </div>
            </Link>
          </li>
        ))}
      </ul>
      {data.pagination.total_pages > 1 && (
        <div className="mt-3">
          <Pagination
            page={data.pagination.page}
            totalPages={data.pagination.total_pages}
            onPage={setPage}
          />
        </div>
      )}
    </div>
  );
}

export function LikersPanel({
  targetType,
  targetId,
  likeCount,
  title = "Likes",
  isOwner = false,
}: {
  targetType: LikeTargetType;
  targetId: string;
  likeCount: number;
  title?: string;
  isOwner?: boolean;
}) {
  return (
    <div className="card">
      <h2 className="stat-label mb-3 border-b border-border pb-2">
        {title} ({likeCount})
      </h2>
      {likeCount > 0 ? (
        <LikersList targetType={targetType} targetId={targetId} />
      ) : (
        <p className="text-muted text-xs">{isOwner ? "No likes yet." : "Be the first to like this."}</p>
      )}
    </div>
  );
}

export function LikesDrawer({
  open,
  onClose,
  targetType,
  targetId,
  likeCount,
  title = "Likes",
}: {
  open: boolean;
  onClose: () => void;
  targetType: LikeTargetType;
  targetId: string;
  likeCount: number;
  title?: string;
}) {
  if (!open) return null;

  return (
    <div className="modal-overlay fixed inset-0 flex justify-end">
      <button
        type="button"
        className="absolute inset-0 cursor-default bg-global-bg/70"
        aria-label="Close likes"
        onClick={onClose}
      />
      <aside
        className="relative flex h-full w-full max-w-sm flex-col border-s-2 border-border bg-global-bg shadow-xl"
        role="dialog"
        aria-label={title}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="title text-base">
            {title} ({likeCount})
          </h2>
          <button type="button" className="btn-secondary px-2 py-1 text-xs" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <LikersList targetType={targetType} targetId={targetId} />
        </div>
      </aside>
    </div>
  );
}

export function ViewLikersTrigger({
  targetType,
  targetId,
  likeCount,
  title,
  className = "",
  emptyLabel = "Be the first to like!",
  isOwner = false,
}: {
  targetType: LikeTargetType;
  targetId: string;
  likeCount: number;
  title?: string;
  className?: string;
  emptyLabel?: string;
  isOwner?: boolean;
}) {
  const [open, setOpen] = useState(false);

  if (likeCount === 0) {
    return (
      <span className={className}>{isOwner ? "No likes yet" : emptyLabel}</span>
    );
  }

  const label = `${likeCount} ${likeCount === 1 ? "like" : "likes"}`;

  return (
    <>
      <button
        type="button"
        className={`hover:text-accent hover:underline ${className}`}
        onClick={() => setOpen(true)}
        title="View who liked this"
      >
        {label}
      </button>
      <LikesDrawer
        open={open}
        onClose={() => setOpen(false)}
        targetType={targetType}
        targetId={targetId}
        likeCount={likeCount}
        title={title}
      />
    </>
  );
}
