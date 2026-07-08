import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DynamicActivityZone, DynamicMetricDistribution, Stream } from "@/types/api";
import { formatDuration } from "@/lib/format";

export function StreamChart({ stream }: { stream: Stream }) {
  const data = stream.data.map((v, i) => ({
    x: stream.axis[i] ?? i,
    value: v,
  }));

  return (
    <div className="h-48 w-full">
      <p className="text-muted text-xs mb-2">{stream.metric_key}</p>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="var(--border)" />
          <XAxis dataKey="x" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} stroke="var(--muted)" />
          <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} stroke="var(--muted)" />
          <Tooltip contentStyle={{ background: "var(--global-bg)", border: "2px solid var(--border)" }} />
          <Line type="monotone" dataKey="value" stroke="var(--accent)" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ZoneChart({ zone }: { zone: DynamicActivityZone }) {
  const total = zone.zones.reduce((s, z) => s + z.time_in_seconds, 0) || 1;
  const colours = ["#94a3b8", "#60a5fa", "#34d399", "#fbbf24", "#f87171", "#c084fc"];

  return (
    <div className="mb-4">
      <p className="text-muted text-xs font-semibold mb-2">{zone.display_name}</p>
      <div className="flex h-6 overflow-hidden border border-border">
        {zone.zones.map((z, i) => (
          <div
            key={z.zone_index}
            style={{
              width: `${(z.time_in_seconds / total) * 100}%`,
              backgroundColor: colours[i % colours.length],
            }}
            title={`Z${z.zone_index}: ${formatDuration(z.time_in_seconds)}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-2 mt-2 text-xs text-muted">
        {zone.zones.map((z) => (
          <span key={z.zone_index}>
            Z{z.zone_index}: {formatDuration(z.time_in_seconds)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function DistributionChart({ dist }: { dist: DynamicMetricDistribution }) {
  const data = dist.buckets.map((b, i) => ({
    name: `${b.min_value}-${b.max_value}`,
    time: b.time_in_seconds,
    index: i,
  }));

  return (
    <div className="h-48 w-full mb-4">
      <p className="text-muted text-xs font-semibold mb-2">{dist.display_name}</p>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="var(--border)" />
          <XAxis dataKey="name" tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }} stroke="var(--muted)" />
          <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} stroke="var(--muted)" />
          <Tooltip contentStyle={{ background: "var(--global-bg)", border: "2px solid var(--border)" }} />
          <Bar dataKey="time" fill="var(--accent)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
