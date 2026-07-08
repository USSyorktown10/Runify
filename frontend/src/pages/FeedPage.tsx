import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { CursorLoader } from "@/components/Pagination";
import { FeedCard } from "@/components/FeedCard";
import { EmptyState } from "@/components/EmptyState";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ClubAvatar } from "@/components/ClubAvatar";
import { useFormatters } from "@/components/MetricGrid";
import { useAuth } from "@/context/AuthContext";
import { athleteName } from "@/lib/format";
import type {
  CursorPaginatedFeedResponse,
  DetailedAthlete,
  PaginatedAthletesResponse,
  PaginatedRoutesResponse,
  PaginatedClubsResponse,
  IntegrationStatus,
} from "@/types/api";

export function FeedPage() {
  const { user } = useAuth();
  const { distance, duration, pace } = useFormatters();

  const { data: feedData, fetchNextPage, hasNextPage, isLoading: feedLoading, isFetchingNextPage } = useInfiniteQuery({
    queryKey: ["feed"],
    queryFn: ({ pageParam }) => {
      const cursor = pageParam ? `&cursor=${encodeURIComponent(pageParam)}` : "";
      return api.get<CursorPaginatedFeedResponse>(`/athlete/feed?limit=20${cursor}`);
    },
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
  });

  // Fetch detailed info of logged-in athlete for stats
  const { data: athlete, isLoading: athleteLoading } = useQuery<DetailedAthlete>({
    queryKey: ["athlete", user?.id],
    queryFn: () => api.get<DetailedAthlete>(`/athletes/${user!.id}`),
    enabled: !!user?.id,
  });

  // Fetch following count
  const { data: following } = useQuery<PaginatedAthletesResponse>({
    queryKey: ["followingCount", user?.id],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/athletes/${user!.id}/following?per_page=1`),
    enabled: !!user?.id,
  });

  // Fetch followers count
  const { data: followers } = useQuery<PaginatedAthletesResponse>({
    queryKey: ["followersCount", user?.id],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/athletes/${user!.id}/followers?per_page=1`),
    enabled: !!user?.id,
  });

  // Fetch routes count
  const { data: routes } = useQuery<PaginatedRoutesResponse>({
    queryKey: ["routesCount", user?.id],
    queryFn: () => api.get<PaginatedRoutesResponse>(`/athletes/${user!.id}/routes?per_page=1`),
    enabled: !!user?.id,
  });

  // Fetch joined clubs
  const { data: clubs } = useQuery<PaginatedClubsResponse>({
    queryKey: ["athleteClubs", user?.id],
    queryFn: () => api.get<PaginatedClubsResponse>(`/athletes/${user!.id}/clubs?per_page=5`),
    enabled: !!user?.id,
  });

  // Fetch device integrations
  const { data: integrations } = useQuery<IntegrationStatus[]>({
    queryKey: ["integrations"],
    queryFn: () => api.get<IntegrationStatus[]>("/integrations"),
    enabled: !!user?.id,
  });

  const feedItems = feedData?.pages.flatMap((p) => p.items) ?? [];
  const location = user ? [user.city, user.state, user.country].filter(Boolean).join(", ") : "";

  return (
    <div className="w-full">
      {/* Page Header (hidden on desktop if we have the profile card, but good for context) */}
      <div className="page-header lg:hidden">
        <h1 className="title">Feed</h1>
        <p className="prose-runify text-muted">Activity and posts from athletes you follow.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN: Athlete profile & stats overview */}
        <aside className="lg:col-span-3 lg:sticky lg:top-20 space-y-4">
          {user && (
            <div className="card overflow-hidden p-0 relative flex flex-col">
              {/* Banner background */}
              <div className="h-16 bg-gradient-to-r from-accent/20 via-accent/30 to-accent/10 border-b border-border" />
              
              {/* Profile main details */}
              <div className="px-4 pb-4 pt-0 flex flex-col items-center -mt-8 relative z-10">
                <div className="border-4 border-global-bg bg-global-bg mb-2 shadow-sm">
                  <AthleteAvatar athlete={user} size="lg" />
                </div>
                <h2 className="subheading text-center line-clamp-1">
                  <Link className="hover:text-accent font-semibold transition-colors" to={`/athletes/${user.id}`}>
                    {athleteName(user)}
                  </Link>
                </h2>
                <p className="meta mt-0.5 text-center">@{user.username}</p>
                {location && <p className="text-xs text-muted mt-1 text-center truncate w-full">{location}</p>}
              </div>
              
              {/* Stats connections */}
              <div className="grid grid-cols-3 border-t border-border bg-surface/50 text-center py-2 text-xs divide-x divide-border">
                <Link to={`/athletes/${user.id}?tab=following`} className="hover:text-accent group py-0.5">
                  <span className="block font-semibold text-sm group-hover:text-accent transition-colors">
                    {following?.pagination.total_items ?? 0}
                  </span>
                  <span className="text-[10px] text-muted uppercase tracking-wider">Following</span>
                </Link>
                <Link to={`/athletes/${user.id}?tab=followers`} className="hover:text-accent group py-0.5">
                  <span className="block font-semibold text-sm group-hover:text-accent transition-colors">
                    {followers?.pagination.total_items ?? 0}
                  </span>
                  <span className="text-[10px] text-muted uppercase tracking-wider">Followers</span>
                </Link>
                <Link to="/routes" className="hover:text-accent group py-0.5">
                  <span className="block font-semibold text-sm group-hover:text-accent transition-colors">
                    {routes?.pagination.total_items ?? 0}
                  </span>
                  <span className="text-[10px] text-muted uppercase tracking-wider">Routes</span>
                </Link>
              </div>
            </div>
          )}

          {/* Quick Actions */}
          <div className="card flex flex-col gap-2">
            <h3 className="stat-label mb-1">Quick Actions</h3>
            <Link to="/activities/new" className="btn-primary text-center text-xs py-2 font-semibold">
              Log Manual Activity
            </Link>
            <Link to="/activities/upload" className="btn-secondary text-center text-xs py-2 font-semibold">
              Upload Activity File
            </Link>
            <Link to="/routes/new" className="btn-secondary text-center text-xs py-2 font-semibold">
              Create Route
            </Link>
          </div>
        </aside>

        {/* MIDDLE COLUMN: Feed */}
        <main className="lg:col-span-6 space-y-4">
          <div className="hidden lg:block border-b-2 border-border pb-3 mb-4">
            <h1 className="title">Feed</h1>
            <p className="prose-runify text-muted">Activity and posts from athletes you follow.</p>
          </div>

          {feedLoading && <p className="text-muted text-center py-8">Loading feed items…</p>}
          
          {!feedLoading && feedItems.length === 0 && (
            <EmptyState title="Your feed is empty" description="Follow athletes to see their activities and posts." />
          )}

          <ul className="flex flex-col gap-4" role="list">
            {feedItems.map((item) => (
              <FeedCard key={item.id} item={item} />
            ))}
          </ul>

          <CursorLoader
            hasMore={!!hasNextPage}
            onLoadMore={() => {
              if (!isFetchingNextPage) fetchNextPage();
            }}
          />
        </main>

        {/* RIGHT COLUMN: Clubs, stats, personal records, integrations */}
        <aside className="lg:col-span-3 space-y-4">
          {/* Stats & Mileage */}
          <div className="card flex flex-col gap-3">
            <h3 className="stat-label border-b border-border pb-1">Your Stats & Mileage</h3>
            {athleteLoading ? (
              <p className="text-xs text-muted">Loading stats…</p>
            ) : athlete ? (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="text-muted block text-[10px] uppercase tracking-wider">YTD Distance</span>
                  <span className="font-semibold text-sm tabular-nums">
                    {distance(athlete.stats.ytd_run_totals)}
                  </span>
                </div>
                <div>
                  <span className="text-muted block text-[10px] uppercase tracking-wider">All-Time Run</span>
                  <span className="font-semibold text-sm tabular-nums">
                    {distance(athlete.stats.all_time_run_totals)}
                  </span>
                </div>
                <div>
                  <span className="text-muted block text-[10px] uppercase tracking-wider">Current FTP</span>
                  <span className="font-semibold text-sm tabular-nums">
                    {athlete.stats.current_ftp > 0 ? `${athlete.stats.current_ftp} W` : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-muted block text-[10px] uppercase tracking-wider">Threshold Pace</span>
                  <span className="font-semibold text-sm tabular-nums">
                    {pace(athlete.stats.threshold_pace)}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-muted">No stats available.</p>
            )}

            {/* Personal Records summary */}
            {athlete && athlete.personal_records.length > 0 && (
              <div className="border-t border-border pt-2 mt-1">
                <h4 className="text-[10px] uppercase text-muted tracking-wider mb-2 font-semibold">Personal Records</h4>
                <ul className="space-y-1">
                  {athlete.personal_records.slice(0, 3).map((pr) => (
                    <li key={pr.distance_name} className="flex justify-between text-xs">
                      <span className="text-muted">{pr.distance_name}</span>
                      <Link className="cactus-link font-semibold transition-colors" to={`/activities/${pr.activity_id}`}>
                        {duration(pr.time_in_seconds)}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Joined Clubs */}
          <div className="card flex flex-col gap-3">
            <div className="flex justify-between items-center border-b border-border pb-1">
              <h3 className="stat-label">My Clubs</h3>
              <Link className="cactus-link text-[11px] font-semibold" to="/clubs">
                Explore
              </Link>
            </div>
            {clubs?.items.length === 0 ? (
              <p className="text-xs text-muted py-1">You haven't joined any clubs yet.</p>
            ) : (
              <ul className="space-y-3">
                {clubs?.items.map((club) => (
                  <li key={club.id} className="flex items-center gap-2">
                    <ClubAvatar club={club} size="sm" />
                    <div className="min-w-0 flex-1">
                      <Link to={`/clubs/${club.id}`} className="hover:text-accent font-semibold text-xs truncate block transition-colors">
                        {club.name}
                      </Link>
                      <span className="text-[10px] text-muted block">
                        {club.member_count} {club.member_count === 1 ? "member" : "members"}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <Link to="/clubs/new" className="btn-secondary text-center text-xs mt-1 py-1.5 font-semibold">
              Create a Club
            </Link>
          </div>

          {/* Integrations (Devices) */}
          <div className="card flex flex-col gap-3">
            <div className="flex justify-between items-center border-b border-border pb-1">
              <h3 className="stat-label">Device Status</h3>
              <Link className="cactus-link text-[11px] font-semibold" to="/settings/integrations">
                Settings
              </Link>
            </div>
            <div className="flex flex-col gap-2 text-xs">
              {["garmin", "apple_health", "wahoo"].map((p) => {
                const isConnected = integrations?.find((i) => i.provider === p)?.is_connected;
                return (
                  <div key={p} className="flex justify-between items-center py-0.5">
                    <span className="capitalize text-muted">{p.replace("_", " ")}</span>
                    {isConnected ? (
                      <span className="inline-flex items-center gap-1 text-accent font-semibold text-[11px]">
                        <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                        Connected
                      </span>
                    ) : (
                      <span className="text-muted/60 text-[11px]">Not Connected</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

