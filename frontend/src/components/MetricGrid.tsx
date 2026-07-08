import { usePreferences } from "@/hooks/usePreferences";
import { formatDate, formatDistance, formatDuration, formatPace } from "@/lib/format";
import type { DynamicWorkoutMetric } from "@/types/api";

export function useFormatters() {
  const { data: prefs } = usePreferences();
  const system = (prefs?.measurement_system === "imperial" ? "imperial" : "metric") as
    | "metric"
    | "imperial";

  return {
    distance: (m: number) => formatDistance(m, system),
    duration: formatDuration,
    pace: (mps: number) => formatPace(mps, system),
    date: formatDate,
    system,
  };
}

export function MetricGrid({ metrics }: { metrics: DynamicWorkoutMetric[] }) {
  if (!metrics.length) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {metrics.map((m) => (
        <span
          key={m.key}
          className="inline-flex items-center gap-1 border border-border rounded-none px-1.5 py-0 text-xs"
        >
          <span className="text-muted">{m.display_name}</span>
          <span className="font-mono font-semibold tabular-nums">
            {m.value} {m.unit}
          </span>
        </span>
      ))}
    </div>
  );
}
