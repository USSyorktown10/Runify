import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ViewLikersTrigger } from "@/components/LikersPanel";
import { Pagination } from "@/components/Pagination";
import { useAuth } from "@/context/AuthContext";
import { athleteName, formatDateTime } from "@/lib/format";
import type { PaginatedCommentsResponse } from "@/types/api";

export type CommentTargetType = "activity" | "post" | "club_post";

function commentsPath(targetType: CommentTargetType, targetId: string): string {
  switch (targetType) {
    case "activity":
      return `/activities/${targetId}/comments`;
    case "post":
      return `/posts/${targetId}/comments`;
    case "club_post":
      return `/club-posts/${targetId}/comments`;
  }
}

export function CommentList({
  targetType,
  targetId,
  variant = "page",
}: {
  targetType: CommentTargetType;
  targetId: string;
  variant?: "page" | "inline";
}) {
  const [page, setPage] = useState(1);
  const [text, setText] = useState("");
  const qc = useQueryClient();
  const { user } = useAuth();

  const isInline = variant === "inline";
  const basePath = commentsPath(targetType, targetId);

  const { data, isLoading } = useQuery({
    queryKey: ["comments", targetType, targetId, page],
    queryFn: () =>
      api.get<PaginatedCommentsResponse>(
        `${basePath}?page=${page}&per_page=${isInline ? 5 : 10}`,
      ),
  });

  const postComment = useMutation({
    mutationFn: (body: string) =>
      api.post(`${basePath}?text=${encodeURIComponent(body)}`),
    onSuccess: () => {
      setText("");
      qc.invalidateQueries({ queryKey: ["comments", targetType, targetId] });
      qc.invalidateQueries({ queryKey: ["feed"] });
      qc.invalidateQueries({ queryKey: ["athlete-feed"] });
      if (targetType === "activity") {
        qc.invalidateQueries({ queryKey: ["activity", targetId] });
      }
    },
  });

  const likeMutation = useMutation({
    mutationFn: ({ commentId, isLiked }: { commentId: string; isLiked: boolean }) => {
      return isLiked
        ? api.delete(`/comments/${commentId}/likes`)
        : api.post(`/comments/${commentId}/likes`);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["comments", targetType, targetId] });
      qc.invalidateQueries({ queryKey: ["likers", "comment"] });
    },
  });

  return (
    <section className={isInline ? "mt-3 pt-3 border-t border-border/40" : "mt-8"}>
      {!isInline && <h2 className="title text-xl mb-4">Comments</h2>}
      <form
        className="mb-4 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) postComment.mutate(text.trim());
        }}
      >
        {user && <AthleteAvatar athlete={user} size="sm" className="shrink-0 mt-1" />}
        <input
          className="field-input flex-1 py-1 text-xs"
          placeholder="Add a comment…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" className="btn-primary py-1 px-3 text-xs cursor-pointer" disabled={postComment.isPending}>
          Post
        </button>
      </form>
      {isLoading && <p className="text-muted text-xs">Loading comments…</p>}
      <ul className={isInline ? "space-y-2 max-h-60 overflow-y-auto" : "space-y-4"}>
        {data?.items.map((c) => {
          const isCommentOwner = user?.id === c.author.id;
          return (
          <li key={c.id} className="flex gap-2.5">
            <Link to={`/athletes/${c.author.id}`} className="shrink-0">
              <AthleteAvatar athlete={c.author} size="sm" />
            </Link>
            <div className="bg-surface border border-border/40 p-2 flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <Link className="font-semibold text-xs truncate hover:text-accent" to={`/athletes/${c.author.id}`}>
                  {athleteName(c.author)}
                </Link>
                <time className="text-muted text-[10px] shrink-0">{formatDateTime(c.created_at)}</time>
              </div>
              <p className="mt-0.5 text-xs whitespace-pre-wrap leading-snug">{c.text}</p>

              <div className="flex items-center gap-2 mt-1.5">
                {!isCommentOwner && (
                  <button
                    onClick={() => likeMutation.mutate({ commentId: c.id, isLiked: c.is_liked })}
                    disabled={likeMutation.isPending}
                    className={`inline-flex items-center gap-1 text-[10px] font-semibold transition-colors hover:text-accent cursor-pointer ${
                      c.is_liked ? "text-accent" : "text-muted"
                    }`}
                    title={c.is_liked ? "Unlike comment" : "Like comment"}
                  >
                    <svg
                      fill={c.is_liked ? "currentColor" : "none"}
                      stroke="currentColor"
                      strokeWidth="1.5"
                      viewBox="0 0 16 16"
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-3.5 h-3.5"
                    >
                      <path d="M6.18.36A.625.625 0 016.746 0h.366a2.625 2.625 0 012.609 2.918L9.374 6h3.69a2.185 2.185 0 011.68 3.584l-.119.142v1.291c0 .458-.16.902-.454 1.254l-.171.205v.399A2.125 2.125 0 0111.875 15H5.703c-.256 0-.507-.077-.72-.22l-1.157-.777a.042.042 0 00-.024-.007l-1.483.031A1.292 1.292 0 011 12.736V8.81c0-.38.168-.742.46-.988l2.032-1.711zm.964.89L4.566 6.765a.625.625 0 01-.163.213l-2.138 1.8a.042.042 0 00-.015.032v3.926c0 .023.02.042.043.041l1.483-.03c.266-.006.527.07.748.219l1.156.777a.042.042 0 00.023.007h6.172a.875.875 0 00.875-.875v-.851l.46-.553a.708.708 0 00.165-.454V9.274l.408-.49a.935.935 0 00-.718-1.534h-5.09l.504-4.471c.09-.805-.53-1.51-1.335-1.529z" />
                    </svg>
                    Like
                  </button>
                )}
                {c.like_count > 0 ? (
                  <ViewLikersTrigger
                    targetType="comment"
                    targetId={c.id}
                    likeCount={c.like_count}
                    title="Comment likes"
                    className="text-[10px] font-semibold text-muted"
                    emptyLabel=""
                    isOwner={isCommentOwner}
                  />
                ) : isCommentOwner ? (
                  <span className="text-[10px] font-semibold text-muted">No likes yet</span>
                ) : null}
              </div>
            </div>
          </li>
        );
        })}
      </ul>
      {data && data.pagination.total_pages > 1 && (
        <div className="mt-3">
          <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />
        </div>
      )}
    </section>
  );
}
