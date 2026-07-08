import { useState, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ClubAvatar } from "@/components/ClubAvatar";
import { RelationshipButton } from "@/components/RelationshipButton";
import { ReportDialog } from "@/components/ReportDialog";
import { useFormatters } from "@/components/MetricGrid";
import { useAuth } from "@/context/AuthContext";
import { athleteName, formatDate } from "@/lib/format";
import { FeedCard } from "@/components/FeedCard";
import { CursorLoader } from "@/components/Pagination";
import { EmptyState } from "@/components/EmptyState";
import { AthleteConnectionList } from "@/pages/athletes/AthleteConnectionList";
import { AthleteSegmentList } from "@/pages/athletes/AthleteSegmentList";
import type {
  DetailedAthlete,
  CursorPaginatedFeedResponse,
  PaginatedAthletesResponse,
  PaginatedClubsResponse,
} from "@/types/api";

type ProfileTabId = "overview" | "following" | "followers" | "segments";

const PROFILE_TABS: ProfileTabId[] = ["overview", "following", "followers", "segments"];

function parseProfileTab(value: string | null): ProfileTabId {
  if (value && PROFILE_TABS.includes(value as ProfileTabId)) {
    return value as ProfileTabId;
  }
  return "overview";
}

function ProfileTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`border-b-2 px-3 py-2 text-xs font-semibold uppercase tracking-wider transition-colors ${
        active
          ? "border-accent text-accent"
          : "border-transparent text-muted hover:text-accent"
      }`}
    >
      {children}
    </button>
  );
}

export function AthleteProfilePage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = parseProfileTab(searchParams.get("tab"));
  const { user } = useAuth();
  const { distance, duration, pace } = useFormatters();
  const [reportOpen, setReportOpen] = useState(false);

  function setActiveTab(tab: ProfileTabId) {
    if (tab === "overview") {
      setSearchParams({}, { replace: true });
    } else {
      setSearchParams({ tab }, { replace: true });
    }
  }

  const { data: athlete, isLoading: athleteLoading } = useQuery({
    queryKey: ["athlete", id],
    queryFn: () => api.get<DetailedAthlete>(`/athletes/${id}`),
    enabled: !!id,
  });

  const { data: following } = useQuery<PaginatedAthletesResponse>({
    queryKey: ["followingCount", id],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/athletes/${id}/following?per_page=1`),
    enabled: !!id,
  });

  const { data: followers } = useQuery<PaginatedAthletesResponse>({
    queryKey: ["followersCount", id],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/athletes/${id}/followers?per_page=1`),
    enabled: !!id,
  });

  const { data: clubs } = useQuery<PaginatedClubsResponse>({
    queryKey: ["athleteClubs", id],
    queryFn: () => api.get<PaginatedClubsResponse>(`/athletes/${id}/clubs?per_page=8`),
    enabled: !!id,
  });

  const { data: feedData, fetchNextPage, hasNextPage, isLoading: feedLoading, isFetchingNextPage } = useInfiniteQuery({
    queryKey: ["athlete-feed", id],
    queryFn: ({ pageParam }) => {
      const cursor = pageParam ? `&cursor=${encodeURIComponent(pageParam)}` : "";
      return api.get<CursorPaginatedFeedResponse>(`/athletes/${id}/feed?limit=20${cursor}`);
    },
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!id && activeTab === "overview",
  });

  if (athleteLoading) return <p className="text-muted">Loading…</p>;
  if (!athlete) return <p className="text-red-500">Athlete not found.</p>;

  const feedItems =
    feedData?.pages
      .flatMap((p) => p.items)
      .filter((item) => item.type === "activity" || item.type === "post") ?? [];
  const location = [athlete.city, athlete.state, athlete.country].filter(Boolean).join(", ");
  const isOwnProfile = user?.id === id;

  return (
    <div className="w-full">
      {/* Profile header — Strava-style banner row */}
      <header className="border-b-2 border-border pb-6 mb-6">
        <div className="mb-5">
          <AthleteAvatar athlete={athlete} size="xl" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          <div className="space-y-2">
            <h1 className="title text-2xl sm:text-3xl" title={`Member since: ${formatDate(athlete.created)}`}>
              {athleteName(athlete)}
            </h1>
            <p className="text-sm text-muted">@{athlete.username}</p>
            {location && (
              <p className="text-sm text-muted flex items-center gap-1.5">
                <span aria-hidden className="text-accent">◎</span>
                {location}
              </p>
            )}
            <p className="text-xs text-muted">Member since {formatDate(athlete.created)}</p>

            {!isOwnProfile && (
              <div className="flex flex-wrap gap-2 pt-2">
                <RelationshipButton athleteId={id!} />
                <button
                  type="button"
                  className="btn-secondary text-xs py-2 px-3 font-semibold"
                  onClick={() => setReportOpen(true)}
                >
                  Report
                </button>
              </div>
            )}
          </div>

          <div className="card">
            <h2 className="stat-label border-b border-border pb-2 mb-3">Summary</h2>
            <div className="grid grid-cols-2 gap-4 text-center sm:grid-cols-3">
              <div>
                <div className="text-xl font-semibold tabular-nums">{distance(athlete.stats.ytd_run_totals)}</div>
                <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5">YTD Distance</div>
              </div>
              <div>
                <div className="text-xl font-semibold tabular-nums">{distance(athlete.stats.all_time_run_totals)}</div>
                <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5">All-Time Run</div>
              </div>
              <div className="col-span-2 sm:col-span-1">
                <div className="text-xl font-semibold tabular-nums">{pace(athlete.stats.threshold_pace)}</div>
                <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5">Threshold Pace</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Main column: tabs + activity/post feed */}
        <main className="lg:col-span-8 space-y-4">
          <nav
            className="flex flex-wrap gap-1 border-b border-border"
            aria-label="Athlete profile sections"
            role="tablist"
          >
            <ProfileTab active={activeTab === "overview"} onClick={() => setActiveTab("overview")}>
              Overview
            </ProfileTab>
            <ProfileTab active={activeTab === "following"} onClick={() => setActiveTab("following")}>
              Following
            </ProfileTab>
            <ProfileTab active={activeTab === "followers"} onClick={() => setActiveTab("followers")}>
              Followers
            </ProfileTab>
            <ProfileTab active={activeTab === "segments"} onClick={() => setActiveTab("segments")}>
              Segments
            </ProfileTab>
          </nav>

          {activeTab === "overview" && (
            <>
              {feedLoading && <p className="text-muted py-8">Loading…</p>}

              {!feedLoading && feedItems.length === 0 && (
                <EmptyState
                  title="No activities yet"
                  description="This athlete hasn't shared any activities or personal posts."
                />
              )}

              <ul className="flex flex-col gap-4" role="list">
                {feedItems.map((item) => (
                  <FeedCard key={`${item.type}-${item.id}`} item={item} />
                ))}
              </ul>

              <CursorLoader
                hasMore={!!hasNextPage}
                onLoadMore={() => {
                  if (!isFetchingNextPage) fetchNextPage();
                }}
              />
            </>
          )}

          {activeTab === "following" && id && <AthleteConnectionList athleteId={id} type="following" />}
          {activeTab === "followers" && id && <AthleteConnectionList athleteId={id} type="followers" />}
          {activeTab === "segments" && id && <AthleteSegmentList athleteId={id} />}
        </main>

        {/* Sidebar */}
        <aside className="lg:col-span-4 space-y-4">
          <div className="card">
            <h3 className="stat-label border-b border-border pb-2 mb-3">Clubs</h3>
            {!clubs?.items.length ? (
              <p className="text-xs text-muted">No joined clubs.</p>
            ) : (
              <ul className="grid grid-cols-4 gap-2">
                {clubs.items.map((club) => (
                  <li key={club.id}>
                    <Link to={`/clubs/${club.id}`} title={club.name} className="block hover:opacity-90">
                      <ClubAvatar club={club} size="md" />
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="card">
            <h3 className="stat-label border-b border-border pb-2 mb-3">Social</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center justify-between">
                <span className="text-muted">Following</span>
                <button
                  type="button"
                  className="font-semibold hover:text-accent tabular-nums"
                  onClick={() => setActiveTab("following")}
                >
                  {following?.pagination.total_items ?? 0}
                </button>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-muted">Followers</span>
                <button
                  type="button"
                  className="font-semibold hover:text-accent tabular-nums"
                  onClick={() => setActiveTab("followers")}
                >
                  {followers?.pagination.total_items ?? 0}
                </button>
              </li>
            </ul>
          </div>

          <div className="card">
            <h3 className="stat-label border-b border-border pb-2 mb-3">Stats</h3>
            <table className="data-table">
              <tbody>
                <tr>
                  <th scope="row">YTD Distance</th>
                  <td className="text-right">{distance(athlete.stats.ytd_run_totals)}</td>
                </tr>
                <tr>
                  <th scope="row">All-Time Run</th>
                  <td className="text-right">{distance(athlete.stats.all_time_run_totals)}</td>
                </tr>
                <tr>
                  <th scope="row">Current FTP</th>
                  <td className="text-right">
                    {athlete.stats.current_ftp > 0 ? `${athlete.stats.current_ftp} W` : "—"}
                  </td>
                </tr>
                <tr>
                  <th scope="row">Threshold Pace</th>
                  <td className="text-right">{pace(athlete.stats.threshold_pace)}</td>
                </tr>
              </tbody>
            </table>

            {athlete.personal_records.length > 0 && (
              <div className="border-t border-border mt-3 pt-3">
                <h4 className="text-[10px] uppercase text-muted tracking-wider mb-2 font-semibold">Personal Records</h4>
                <table className="data-table">
                  <tbody>
                    {athlete.personal_records.map((pr) => (
                      <tr key={pr.distance_name}>
                        <th scope="row">{pr.distance_name}</th>
                        <td className="text-right">
                          <Link className="cactus-link font-semibold" to={`/activities/${pr.activity_id}`}>
                            {duration(pr.time_in_seconds)}
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </aside>
      </div>

      <ReportDialog open={reportOpen} targetType="athlete" targetId={id!} onClose={() => setReportOpen(false)} />
    </div>
  );
}
