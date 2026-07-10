import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ClubAvatar } from "@/components/ClubAvatar";
import { athleteName } from "@/lib/format";
import type {
  PaginatedClubsResponse,
  PaginatedConnectionsResponse,
  PaginatedSegmentsResponse,
} from "@/types/api";

export function SearchModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"athletes" | "segments" | "clubs">("athletes");

  const { data: athletes } = useQuery({
    queryKey: ["search-athletes", query],
    queryFn: () =>
      api.get<PaginatedConnectionsResponse>(`/athletes/search?query=${encodeURIComponent(query)}&per_page=5`),
    enabled: open && tab === "athletes" && query.length > 0,
  });

  const { data: segments } = useQuery({
    queryKey: ["search-segments", query],
    queryFn: () =>
      api.get<PaginatedSegmentsResponse>(`/segments?query=${encodeURIComponent(query)}&per_page=5`),
    enabled: open && tab === "segments" && query.length > 0,
  });

  const { data: clubs } = useQuery({
    queryKey: ["search-clubs", query],
    queryFn: () => api.get<PaginatedClubsResponse>(`/clubs?query=${encodeURIComponent(query)}&per_page=5`),
    enabled: open && tab === "clubs" && query.length > 0,
  });

  if (!open) return null;

  return (
    <div className="modal-overlay fixed inset-0 flex items-start justify-center bg-global-bg/90 p-4 pt-24">
      <div className="card w-full max-w-lg border-2 bg-global-bg" role="dialog" aria-label="Search">
        <div className="flex gap-2 mb-4">
          {(["athletes", "segments", "clubs"] as const).map((t) => (
            <button
              key={t}
              type="button"
              className={`text-sm font-semibold  ${tab === t ? "text-accent underline" : "text-muted"}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </div>
        <input
          className="field-input mb-4"
          type="search"
          placeholder="Search…"
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <ul className="space-y-2 max-h-64 overflow-y-auto">
          {tab === "athletes" &&
            athletes?.items.map((r) => (
              <li key={r.athlete.id}>
                <Link
                  className="flex items-center gap-2 hover:text-accent"
                  to={`/athletes/${r.athlete.id}`}
                  onClick={onClose}
                >
                  <AthleteAvatar athlete={r.athlete} size="sm" />
                  <span className="cactus-link">{athleteName(r.athlete)}</span>
                </Link>
              </li>
            ))}
          {tab === "segments" &&
            segments?.items.map((s) => (
              <li key={s.id}>
                <Link className="cactus-link" to={`/segments/${s.id}`} onClick={onClose}>
                  {s.name}
                </Link>
              </li>
            ))}
          {tab === "clubs" &&
            clubs?.items.map((c) => (
              <li key={c.id}>
                <Link className="flex items-center gap-2 hover:text-accent" to={`/clubs/${c.id}`} onClick={onClose}>
                  <ClubAvatar club={c} size="sm" />
                  <span className="cactus-link">{c.name}</span>
                </Link>
              </li>
            ))}
        </ul>
        <button type="button" className="btn-secondary mt-4 w-full" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
