export function formatDistance(meters: number, system: "metric" | "imperial" = "metric"): string {
  if (system === "imperial") {
    const miles = meters / 1609.344;
    return miles >= 0.1 ? `${miles.toFixed(2)} mi` : `${(meters * 3.28084).toFixed(0)} ft`;
  }
  if (meters >= 1000) return `${(meters / 1000).toFixed(2)} km`;
  return `${meters.toFixed(0)} m`;
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatPace(metersPerSecond: number, system: "metric" | "imperial" = "metric"): string {
  if (!metersPerSecond) return "—";
  if (system === "imperial") {
    const secPerMile = 1609.344 / metersPerSecond;
    const m = Math.floor(secPerMile / 60);
    const s = Math.round(secPerMile % 60);
    return `${m}:${String(s).padStart(2, "0")} /mi`;
  }
  const secPerKm = 1000 / metersPerSecond;
  const m = Math.floor(secPerKm / 60);
  const s = Math.round(secPerKm % 60);
  return `${m}:${String(s).padStart(2, "0")} /km`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function athleteName(a: { first_name: string; last_name: string }): string {
  return `${a.first_name} ${a.last_name}`.trim();
}
