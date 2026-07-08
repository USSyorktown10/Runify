import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { MapView } from "@/components/MapView";
import { useFormatters } from "@/components/MetricGrid";
import { useAuth } from "@/context/AuthContext";
import { BackButton } from "@/components/BackButton";
import type { DetailedRoute } from "@/types/api";

export function RouteDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const { distance } = useFormatters();

  const { data: route, isLoading } = useQuery({
    queryKey: ["route", id],
    queryFn: () => api.get<DetailedRoute>(`/routes/${id}`),
    enabled: !!id,
  });

  const download = async (format: string) => {
    const blob = await api.download(`/routes/${id}/export?format=${format}`);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `route.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) return <p className="text-muted">Loading…</p>;
  if (!route) return <p className="text-red-500">Route not found.</p>;

  const isOwner = user?.id === route.athlete_id;

  return (
    <section>
      <BackButton label="Back" />
      <div className="flex justify-between items-start mb-6">
        <h1 className="title text-2xl">{route.name}</h1>
        {isOwner && (
          <Link className="btn-secondary text-sm" to={`/routes/${id}/edit`}>
            Edit
          </Link>
        )}
      </div>
      <p className="text-muted mb-4">{distance(route.distance)} · +{route.elevation_gain.toFixed(0)} m</p>
      {route.description && <p className="mb-6">{route.description}</p>}
      {route.polyline && <MapView polyline={route.polyline} className="h-72 mb-8" />}
      <div className="flex gap-2">
        <button type="button" className="btn-secondary text-sm" onClick={() => download("gpx")}>
          Export GPX
        </button>
        <button type="button" className="btn-secondary text-sm" onClick={() => download("fit")}>
          Export FIT
        </button>
      </div>
    </section>
  );
}
