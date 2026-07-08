import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ClubPostCard } from "@/components/FeedCard";
import { Pagination } from "@/components/Pagination";
import { EmptyState } from "@/components/EmptyState";
import type { PaginatedPostsResponse } from "@/types/api";

export function ClubPostsTab({ clubId, canPost }: { clubId: string; canPost: boolean }) {
  const [page, setPage] = useState(1);
  const [postTitle, setPostTitle] = useState("");
  const [postBody, setPostBody] = useState("");
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["club-posts", clubId, page],
    queryFn: () => api.get<PaginatedPostsResponse>(`/clubs/${clubId}/posts?page=${page}&per_page=10`),
    enabled: !!clubId,
  });

  const createPost = useMutation({
    mutationFn: () => api.post(`/clubs/${clubId}/posts`, { title: postTitle, body: postBody }),
    onSuccess: () => {
      setPostTitle("");
      setPostBody("");
      qc.invalidateQueries({ queryKey: ["club-posts", clubId] });
    },
  });

  return (
    <div>
      {canPost && (
        <form
          className="card mb-6 space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            createPost.mutate();
          }}
        >
          <h2 className="title text-base">Create post</h2>
          <input
            className="field-input"
            placeholder="Title"
            value={postTitle}
            onChange={(e) => setPostTitle(e.target.value)}
            required
          />
          <textarea
            className="field-input min-h-20"
            placeholder="What's on your mind?"
            value={postBody}
            onChange={(e) => setPostBody(e.target.value)}
            required
          />
          <button type="submit" className="btn-primary text-sm" disabled={createPost.isPending}>
            Post
          </button>
        </form>
      )}

      {isLoading && <p className="text-muted">Loading posts…</p>}
      {!isLoading && data?.items.length === 0 && (
        <EmptyState title="No posts yet" description="Club posts from members will appear here." />
      )}
      <ul className="space-y-4">
        {data?.items.map((post) => (
          <li key={post.id} className="list-none">
            <ClubPostCard clubPost={post} athlete={post.author} createdAt={post.created_at} />
          </li>
        ))}
      </ul>
      {data && data.pagination.total_pages > 1 && (
        <Pagination page={data.pagination.page} totalPages={data.pagination.total_pages} onPage={setPage} />
      )}
    </div>
  );
}
