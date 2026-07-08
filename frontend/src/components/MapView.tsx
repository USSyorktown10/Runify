import { useEffect } from "react";
import { MapContainer, Polyline, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { decodePolyline } from "@/lib/polyline";

function FitBounds({ positions }: { positions: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (positions.length > 1) {
      map.fitBounds(positions, { padding: [24, 24] });
    } else if (positions.length === 1) {
      map.setView(positions[0], 14);
    }
  }, [map, positions]);
  return null;
}

export function MapView({
  polyline,
  className = "h-64 w-full rounded-none",
  interactive = true,
}: {
  polyline: string;
  className?: string;
  interactive?: boolean;
}) {
  const positions = decodePolyline(polyline);
  const center: [number, number] = positions[0] ?? [51.505, -0.09];

  return (
    <div className={`map-view relative isolate z-0 ${className}`}>
      <MapContainer
        center={center}
        zoom={13}
        className="h-full w-full rounded-none"
        scrollWheelZoom={interactive}
        zoomControl={interactive}
        dragging={interactive}
        doubleClickZoom={interactive}
        touchZoom={interactive}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {positions.length > 0 && (
          <>
            <Polyline positions={positions} pathOptions={{ color: "#2abc89", weight: 4 }} />
            <FitBounds positions={positions} />
          </>
        )}
      </MapContainer>
    </div>
  );
}
