import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api/client";
import { MapView } from "@/components/MapView";
import { useFormatters } from "@/components/MetricGrid";
import { BackButton } from "@/components/BackButton";
import type { DetailedSegment } from "@/types/api";

export function SegmentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();
  const { distance } = useFormatters();

  const { data: segment, isLoading } = useQuery({
    queryKey: ["segment", id],
    queryFn: () => api.get<DetailedSegment>(`/segments/${id}`),
    enabled: !!id,
  });

  const star = useMutation({
    mutationFn: () =>
      segment?.is_starred ? api.delete(`/segments/${id}/star`) : api.post(`/segments/${id}/star`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["segment", id] }),
  });

  if (isLoading) return <p className="text-muted">Loading…</p>;
  if (!segment) return <p className="text-red-500">Segment not found.</p>;

  return (
    <section>
      <BackButton label="Back" />
      <div className="flex justify-between items-start mb-6">
        <h1 className="title text-2xl">{segment.name}</h1>
        <button type="button" className="btn-secondary" onClick={() => star.mutate()}>
          {segment.is_starred ? "Unstar" : "Star"}
        </button>
      </div>
      <p className="text-muted mb-4">
        {distance(segment.distance)} · {segment.total_athlete_count} athletes · {segment.total_effort_count} efforts
      </p>
      {segment.polyline && (
        <MapView polyline={segment.polyline} className="mb-4 h-[24rem] w-full rounded-none border border-border lg:h-[32rem]" />
      )}
      <nav className="flex gap-4">
        <Link className="cactus-link" to={`/segments/${id}/leaderboard`}>
          Leaderboard
        </Link>
        <Link className="cactus-link" to={`/segments/${id}/efforts`}>
          My efforts
        </Link>
      </nav>
    </section>
  );
}
