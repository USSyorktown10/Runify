import { useEffect, useRef } from "react";

export function CursorLoader({ onLoadMore, hasMore }: { onLoadMore: () => void; hasMore: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!hasMore) return;
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) onLoadMore();
      },
      { rootMargin: "200px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [hasMore, onLoadMore]);

  if (!hasMore) return null;
  return <div ref={ref} className="h-8 text-center text-muted py-4" aria-hidden="true" />;
}

export function Pagination({
  page,
  totalPages,
  onPage,
}: {
  page: number;
  totalPages: number;
  onPage: (p: number) => void;
}) {
  if (totalPages <= 1) return null;
  return (
    <div className="flex gap-2 justify-center mt-8">
      <button
        type="button"
        className="btn-secondary"
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
      >
        Previous
      </button>
      <span className="py-2 text-muted">
        {page} / {totalPages}
      </span>
      <button
        type="button"
        className="btn-secondary"
        disabled={page >= totalPages}
        onClick={() => onPage(page + 1)}
      >
        Next
      </button>
    </div>
  );
}
