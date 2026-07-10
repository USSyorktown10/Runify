import { useState, useRef, useEffect } from "react";
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

export function HeaderSearch() {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"athletes" | "segments" | "clubs">("athletes");
  const [focused, setFocused] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setFocused(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const { data: athletes, isFetching: loadingAthletes } = useQuery({
    queryKey: ["search-athletes", query],
    queryFn: () =>
      api.get<PaginatedConnectionsResponse>(
        `/athletes/search?query=${encodeURIComponent(query)}&per_page=5`
      ),
    enabled: focused && tab === "athletes" && query.length > 0,
  });

  const { data: segments, isFetching: loadingSegments } = useQuery({
    queryKey: ["search-segments", query],
    queryFn: () =>
      api.get<PaginatedSegmentsResponse>(
        `/segments?query=${encodeURIComponent(query)}&per_page=5`
      ),
    enabled: focused && tab === "segments" && query.length > 0,
  });

  const { data: clubs, isFetching: loadingClubs } = useQuery({
    queryKey: ["search-clubs", query],
    queryFn: () =>
      api.get<PaginatedClubsResponse>(
        `/clubs?query=${encodeURIComponent(query)}&per_page=5`
      ),
    enabled: focused && tab === "clubs" && query.length > 0,
  });

  const loading =
    (tab === "athletes" && loadingAthletes) ||
    (tab === "segments" && loadingSegments) ||
    (tab === "clubs" && loadingClubs);

  const hasResults =
    (tab === "athletes" && (athletes?.items.length ?? 0) > 0) ||
    (tab === "segments" && (segments?.items.length ?? 0) > 0) ||
    (tab === "clubs" && (clubs?.items.length ?? 0) > 0);

  return (
    <div ref={containerRef} className="relative w-40 sm:w-48 md:w-60 lg:w-72">
      <div className="relative">
        <input
          className="w-full rounded-none border border-border bg-surface/50 py-1.5 pl-8 pr-3 text-xs focus:border-global-text focus:bg-surface focus:outline-none transition-all duration-150"
          type="search"
          placeholder="Search..."
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setFocused(true);
          }}
          onFocus={() => setFocused(true)}
        />
        <svg
          className="absolute left-3 top-2.5 h-3.5 w-3.5 text-muted pointer-events-none"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2.5}
        >
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>
      </div>

      {focused && (
        <div className="absolute start-0 top-full z-50 mt-2 w-72 sm:w-80 md:w-96 rounded-none bg-surface shadow-[0_12px_40px_rgba(0,0,0,0.08)] p-4 border border-border/50 animate-fade-in-up">
          <div className="flex gap-1.5 mb-3 border-b border-border/40 pb-2">
            {(["athletes", "segments", "clubs"] as const).map((t) => (
              <button
                key={t}
                type="button"
                className={`text-xs font-bold px-3.5 py-1.5 rounded-none transition-all duration-200 cursor-pointer ${
                  tab === t
                    ? "bg-accent text-global-bg"
                    : "text-muted hover:bg-zinc-100 hover:text-global-text dark:hover:bg-zinc-800"
                }`}
                onClick={() => setTab(t)}
              >
                {t}
              </button>
            ))}
          </div>

          <div className="max-h-64 overflow-y-auto pr-1">
            {query.length === 0 ? (
              <p className="text-xs text-muted py-2 text-center">
                Type to search athletes, segments, and clubs...
              </p>
            ) : loading ? (
              <p className="text-xs text-muted py-2 text-center">Searching...</p>
            ) : !hasResults ? (
              <p className="text-xs text-muted py-2 text-center">No results found.</p>
            ) : (
              <ul className="space-y-2">
                {tab === "athletes" &&
                  athletes?.items.map((r) => (
                    <li key={r.athlete.id}>
                      <Link
                        className="flex items-center gap-2 rounded-none p-1 hover:bg-global-bg transition-colors"
                        to={`/athletes/${r.athlete.id}`}
                        onClick={() => setFocused(false)}
                      >
                        <AthleteAvatar athlete={r.athlete} size="sm" />
                        <span className="text-xs font-medium text-global-text hover:text-accent">
                          {athleteName(r.athlete)}
                        </span>
                      </Link>
                    </li>
                  ))}
                {tab === "segments" &&
                  segments?.items.map((s) => (
                    <li key={s.id}>
                      <Link
                        className="block rounded-none p-2 text-xs font-medium text-global-text hover:text-accent hover:bg-global-bg transition-colors"
                        to={`/segments/${s.id}`}
                        onClick={() => setFocused(false)}
                      >
                        {s.name}
                      </Link>
                    </li>
                  ))}
                {tab === "clubs" &&
                  clubs?.items.map((c) => (
                    <li key={c.id}>
                      <Link
                        className="flex items-center gap-2 rounded-none p-1 hover:bg-global-bg transition-colors"
                        to={`/clubs/${c.id}`}
                        onClick={() => setFocused(false)}
                      >
                        <ClubAvatar club={c} size="sm" />
                        <span className="text-xs font-medium text-global-text hover:text-accent">
                          {c.name}
                        </span>
                      </Link>
                    </li>
                  ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
