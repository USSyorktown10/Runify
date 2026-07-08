import polyline from "@mapbox/polyline";

export function decodePolyline(encoded: string): [number, number][] {
  if (!encoded) return [];
  try {
    return polyline.decode(encoded).map(([lat, lng]) => [lat, lng] as [number, number]);
  } catch {
    return [];
  }
}
