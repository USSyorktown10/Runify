import { useState, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DetailedActivity, Stream } from "@/types/api";

function MetricChartTooltip({ active, payload, config }: any) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-global-bg/95 border border-border px-3 py-1 text-[11px] font-semibold text-slate-800 dark:text-slate-200 shadow-none backdrop-blur-md rounded-none flex items-center gap-1.5 font-mono pointer-events-none z-50">
        <span className="w-1.5 h-1.5 rounded-none" style={{ backgroundColor: config.color }} />
        <span>{config.label}:</span>
        <strong className="font-extrabold text-[#0f172a] dark:text-white">
          {config.format(payload[0].value)}
        </strong>
      </div>
    );
  }
  return null;
}

interface ActivityAnalysisProps {
  streams: Stream[];
  activity: DetailedActivity;
  system: "metric" | "imperial";
  onHoverIndexChange?: (index: number | null) => void;
}

interface ChartDataPoint {
  index: number;
  distance: number;
  time: number;
  heart_rate?: number;
  altitude?: number;
  pace?: number;
  cadence?: number;
  power?: number;
  grade_adjusted_pace?: number;
  lat?: number;
  lng?: number;
}

export function ActivityAnalysis({
  streams,
  activity,
  system,
  onHoverIndexChange,
}: ActivityAnalysisProps) {
  const [xAxisType, setXAxisType] = useState<"distance" | "time">("distance");
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const isCycling = ["ride", "cycling", "ebikeride", "virtualride"].includes(
    activity.activity_type.toLowerCase()
  );

  const { chartData, availableMetrics } = useMemo(() => {
    if (!streams || streams.length === 0) {
      return { chartData: [], availableMetrics: [] };
    }

    const maxLenStream = streams.reduce(
      (max, s) => (s.data?.length > (max?.data?.length || 0) ? s : max),
      streams[0]
    );

    if (!maxLenStream || !maxLenStream.data) {
      return { chartData: [], availableMetrics: [] };
    }

    const length = maxLenStream.data.length;
    let processedStreams = [...streams];

    const hasPaceStream = streams.some(
      (s) => s.metric_key === "pace" || s.metric_key === "speed" || s.metric_key === "velocity"
    );

    if (!hasPaceStream) {
      const distStream = streams.find((s) => s.metric_key === "distance");
      const timeStream = streams.find((s) => s.metric_key === "time") || maxLenStream;
      if (distStream && timeStream) {
        const paceData: number[] = [0];
        for (let j = 1; j < length; j++) {
          const d1 = distStream.data[j];
          const d0 = distStream.data[j - 1];
          const t1 = timeStream.axis ? timeStream.axis[j] : j;
          const t0 = timeStream.axis ? timeStream.axis[j - 1] : j - 1;
          const dt = t1 - t0;
          const speed = dt > 0 ? (d1 - d0) / dt : 0;
          paceData.push(Math.min(speed, 25));
        }
        processedStreams.push({
          metric_key: "pace",
          stream_type: "calculated",
          data: paceData,
          axis: distStream.axis,
          axis_type: "distance",
        });
      }
    }

    const dataPoints: ChartDataPoint[] = [];
    for (let i = 0; i < length; i++) {
      const pt: ChartDataPoint = { index: i, distance: 0, time: 0 };
      processedStreams.forEach((s) => {
        const idx = Math.min(i, s.data.length - 1);
        const val = s.data[idx];
        if (s.metric_key === "heartrate" || s.metric_key === "heart_rate") {
          pt.heart_rate = val;
        } else if (s.metric_key === "altitude") {
          pt.altitude = val;
        } else if (
          s.metric_key === "pace" ||
          s.metric_key === "speed" ||
          s.metric_key === "velocity"
        ) {
          pt.pace = val;
        } else if (s.metric_key === "cadence") {
          pt.cadence = val;
        } else if (s.metric_key === "power") {
          pt.power = val;
        } else if (s.metric_key === "grade_adjusted_pace") {
          pt.grade_adjusted_pace = val;
        } else if (s.metric_key === "lat") {
          pt.lat = val;
        } else if (s.metric_key === "lng") {
          pt.lng = val;
        }
        if (s.axis && s.axis[idx] !== undefined) {
          if (s.axis_type === "distance") {
            pt.distance = s.axis[idx];
          } else if (s.axis_type === "time") {
            pt.time = s.axis[idx];
          }
        }
      });
      if (pt.time === 0 && i > 0) {
        pt.time = i * (activity.moving_time / length);
      }
      if (pt.distance === 0 && i > 0) {
        pt.distance = i * (activity.distance / length);
      }
      dataPoints.push(pt);
    }

    const keys = new Set<string>();
    processedStreams.forEach((s) => {
      let key = s.metric_key;
      if (key === "heartrate") key = "heart_rate";
      if (key === "speed" || key === "velocity") key = "pace";
      if (
        ["heart_rate", "altitude", "pace", "cadence", "power", "grade_adjusted_pace"].includes(key)
      ) {
        keys.add(key);
      }
    });

    return { chartData: dataPoints, availableMetrics: Array.from(keys) };
  }, [streams, activity, isCycling]);

  const [visibleMetrics, setVisibleMetrics] = useState<string[]>(() => {
    const defaults = ["pace", "heart_rate", "altitude"];
    return availableMetrics.filter((m) => defaults.includes(m));
  });

  const averages = useMemo(() => {
    const avgs: Record<string, number> = {};
    availableMetrics.forEach((key) => {
      const vals = chartData
        .map((d) => d[key as keyof ChartDataPoint] as number)
        .filter((v) => v !== undefined && v > 0);
      avgs[key] = vals.length === 0 ? 0 : vals.reduce((a, b) => a + b, 0) / vals.length;
    });
    return avgs;
  }, [chartData, availableMetrics]);

  const toggleMetric = (metric: string) => {
    setVisibleMetrics((prev) =>
      prev.includes(metric) ? prev.filter((m) => m !== metric) : [...prev, metric]
    );
  };

  const formatXVal = (pt: ChartDataPoint) => {
    if (xAxisType === "distance") {
      const dist = pt.distance;
      if (system === "imperial") {
        return `${(dist / 1609.344).toFixed(1)} mi`;
      }
      return `${(dist / 1000).toFixed(1)} km`;
    } else {
      const sec = pt.time;
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
      return `${m}:${String(s).padStart(2, "0")}`;
    }
  };

  const formatPaceVal = (mps: number) => {
    if (!mps || mps <= 0.1) return "—";
    if (isCycling) {
      const speed = system === "imperial" ? mps * 2.23694 : mps * 3.6;
      return `${speed.toFixed(1)} ${system === "imperial" ? "mph" : "km/h"}`;
    } else {
      const secPerUnit = system === "imperial" ? 1609.344 / mps : 1000 / mps;
      const m = Math.floor(secPerUnit / 60);
      const s = Math.round(secPerUnit % 60);
      return `${m}:${String(s).padStart(2, "0")} /${system === "imperial" ? "mi" : "km"}`;
    }
  };

  const getMetricConfig = (key: string) => {
    switch (key) {
      case "heart_rate":
        return {
          label: "Heart Rate",
          color: "var(--accent)",
          fillColor: "rgba(16, 185, 129, 0.1)",
          unit: "bpm",
          format: (val: number) => `${val.toFixed(0)} bpm`,
        };
      case "altitude":
        return {
          label: "Elevation",
          color: "var(--phrase-longer, #d97706)",
          fillColor: "rgba(217, 119, 6, 0.1)",
          unit: system === "imperial" ? "ft" : "m",
          format: (val: number) =>
            system === "imperial"
              ? `${(val * 3.28084).toFixed(0)} ft`
              : `${val.toFixed(0)} m`,
        };
      case "pace":
        return {
          label: isCycling ? "Speed" : "Pace",
          color: "var(--accent)",
          fillColor: "rgba(16, 185, 129, 0.1)",
          unit: isCycling
            ? system === "imperial"
              ? "mph"
              : "km/h"
            : system === "imperial"
            ? "/mi"
            : "/km",
          format: formatPaceVal,
        };
      case "cadence":
        return {
          label:
            activity.activity_type.toLowerCase() === "run" ? "Stride Cadence" : "Cadence",
          color: "var(--phrase-more, #4f46e5)",
          fillColor: "rgba(79, 70, 229, 0.1)",
          unit: activity.activity_type.toLowerCase() === "run" ? "spm" : "rpm",
          format: (val: number) =>
            activity.activity_type.toLowerCase() === "run"
              ? `${(val * 2).toFixed(0)} spm`
              : `${val.toFixed(0)} rpm`,
        };
      case "power":
        return {
          label: "Power",
          color: "var(--phrase-longer, #d97706)",
          fillColor: "rgba(217, 119, 6, 0.1)",
          unit: "W",
          format: (val: number) => `${val.toFixed(0)} W`,
        };
      case "grade_adjusted_pace":
        return {
          label: "Grade Adjusted Pace",
          color: "var(--phrase-together, #db2777)",
          fillColor: "rgba(219, 39, 119, 0.1)",
          unit: system === "imperial" ? "/mi" : "/km",
          format: formatPaceVal,
        };
      default:
        return {
          label: key,
          color: "var(--muted)",
          fillColor: "rgba(148, 163, 184, 0.1)",
          unit: "",
          format: (val: number) => val.toString(),
        };
    }
  };

  const activePoint = hoverIndex !== null ? chartData[hoverIndex] : null;

  const handleMouseMove = (e: any) => {
    if (e && e.activeTooltipIndex !== undefined) {
      setHoverIndex(e.activeTooltipIndex);
      if (onHoverIndexChange) onHoverIndexChange(e.activeTooltipIndex);
    }
  };

  const handleMouseLeave = () => {
    setHoverIndex(null);
    if (onHoverIndexChange) onHoverIndexChange(null);
  };

  if (chartData.length === 0) {
    return (
      <div className="card text-center text-muted py-8">
        No sensor analysis streams available for this activity.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Dynamic Hover Stats Bar */}
      <div className="card grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 bg-surface/40 divide-y md:divide-y-0 md:divide-x divide-border">
        <div className="flex flex-col">
          <span className="meta font-semibold tracking-wider">Position</span>
          <span className="stat mt-1">{activePoint ? formatXVal(activePoint) : "—"}</span>
        </div>
        {availableMetrics.map((key) => {
          const config = getMetricConfig(key);
          const rawVal = activePoint
            ? (activePoint[key as keyof ChartDataPoint] as number)
            : undefined;
          const avgVal = averages[key] ?? 0;
          return (
            <div key={key} className="flex flex-col md:pl-4">
              <span className="meta font-semibold tracking-wider" style={{ color: config.color }}>
                {config.label}
              </span>
              <span className="stat mt-1">
                {rawVal !== undefined ? config.format(rawVal) : config.format(avgVal)}
              </span>
            </div>
          );
        })}
      </div>

      {/* Metric Selector + X-Axis Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap gap-1">
          {availableMetrics.map((key) => {
            const config = getMetricConfig(key);
            const isVisible = visibleMetrics.includes(key);
            return (
              <button
                key={key}
                onClick={() => toggleMetric(key)}
                className={`px-2.5 py-1 rounded-none border text-xs font-semibold transition-all cursor-pointer ${
                  isVisible
                    ? "bg-accent/10 border-accent/40 text-accent"
                    : "border-border bg-surface text-muted hover:text-global-text hover:border-muted"
                }`}
              >
                <span
                  className="inline-block w-1.5 h-1.5 rounded-none me-1.5"
                  style={{ backgroundColor: config.color }}
                />
                {config.label}
              </button>
            );
          })}
        </div>

        <div className="flex bg-surface p-0.5 rounded-none border border-border">
          <button
            onClick={() => setXAxisType("distance")}
            className={`px-2 py-0.5 rounded-none text-xs font-semibold transition-all cursor-pointer ${
              xAxisType === "distance"
                ? "bg-global-bg text-global-text shadow-none"
                : "text-muted hover:text-global-text"
            }`}
          >
            Distance
          </button>
          <button
            onClick={() => setXAxisType("time")}
            className={`px-2 py-0.5 rounded-none text-xs font-semibold transition-all cursor-pointer ${
              xAxisType === "time"
                ? "bg-global-bg text-global-text shadow-none"
                : "text-muted hover:text-global-text"
            }`}
          >
            Time
          </button>
        </div>
      </div>

      {/* Stacked Charts */}
      <div className="space-y-4">
        {visibleMetrics.map((key, index) => {
          const config = getMetricConfig(key);
          const isBottomChart = index === visibleMetrics.length - 1;
          return (
            <div key={key} className="card relative p-0 overflow-hidden">
              <div className="p-3 border-b border-border bg-surface/50 flex items-center gap-2">
                <span className="w-2 h-2 rounded-none" style={{ backgroundColor: config.color }} />
                <span className="text-xs font-bold tracking-wider text-muted">
                  {config.label}
                </span>
              </div>
              <div className="h-44 w-full p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={chartData}
                    syncId="activity-analysis-sync"
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                    margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                  >
                    <defs>
                      <linearGradient id={`grad-${key}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={config.color} stopOpacity={0.15} />
                        <stop offset="95%" stopColor={config.color} stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      stroke="var(--border)"
                      strokeDasharray="3 3"
                      vertical={false}
                    />
                    <XAxis
                      dataKey={xAxisType}
                      hide={!isBottomChart}
                      tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
                      stroke="var(--muted)"
                      tickFormatter={(val) => {
                        if (xAxisType === "distance") {
                          if (system === "imperial") return `${(val / 1609.344).toFixed(1)} mi`;
                          return `${(val / 1000).toFixed(1)} km`;
                        } else {
                          const m = Math.floor(val / 60);
                          return `${m}m`;
                        }
                      }}
                    />
                    <YAxis
                      domain={["auto", "auto"]}
                      tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
                      stroke="var(--muted)"
                      width={45}
                      tickFormatter={(val) => {
                        if (key === "pace" || key === "grade_adjusted_pace") {
                          if (isCycling) return val.toFixed(0);
                          if (val <= 0.1) return "—";
                          const secPerUnit =
                            system === "imperial" ? 1609.344 / val : 1000 / val;
                          const m = Math.floor(secPerUnit / 60);
                          const s = Math.round(secPerUnit % 60);
                          return `${m}:${String(s).padStart(2, "0")}`;
                        }
                        return val.toFixed(0);
                      }}
                    />
                    <Tooltip content={<MetricChartTooltip config={config} />} />
                    <Area
                      type="step"
                      dataKey={key}
                      stroke={config.color}
                      strokeWidth={1.5}
                      fillOpacity={1}
                      fill={`url(#grad-${key})`}
                      dot={false}
                      activeDot={{ r: 4, strokeWidth: 1 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
