import { useState, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ClubAvatar } from "@/components/ClubAvatar";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ReportDialog } from "@/components/ReportDialog";
import { useAuth } from "@/context/AuthContext";
import { athleteName } from "@/lib/format";
import { ClubLeaderboardTab } from "@/pages/clubs/ClubLeaderboardTab";
import { ClubRecentActivityTab } from "@/pages/clubs/ClubRecentActivityTab";
import { ClubMembersTab } from "@/pages/clubs/ClubMembersTab";
import { ClubPostsTab } from "@/pages/clubs/ClubPostsTab";
import type {
  DetailedClub,
  PaginatedAthletesResponse,
  PaginatedConnectionsResponse,
} from "@/types/api";

type ClubTabId = "leaderboard" | "activity" | "members" | "posts";

const CLUB_TABS: ClubTabId[] = ["leaderboard", "activity", "members", "posts"];

function parseClubTab(value: string | null): ClubTabId {
  if (value && CLUB_TABS.includes(value as ClubTabId)) {
    return value as ClubTabId;
  }
  return "leaderboard";
}

function ClubTab({
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
        active ? "border-accent text-accent" : "border-transparent text-muted hover:text-accent"
      }`}
    >
      {children}
    </button>
  );
}

export function ClubDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = parseClubTab(searchParams.get("tab"));
  const { user } = useAuth();
  const qc = useQueryClient();
  const [reportOpen, setReportOpen] = useState(false);
  const [inviteQuery, setInviteQuery] = useState("");
  const [inviteMessage, setInviteMessage] = useState<string | null>(null);

  function setActiveTab(tab: ClubTabId) {
    if (tab === "leaderboard") {
      setSearchParams({}, { replace: true });
    } else {
      setSearchParams({ tab }, { replace: true });
    }
  }

  const { data: club, isLoading } = useQuery({
    queryKey: ["club", id],
    queryFn: () => api.get<DetailedClub>(`/clubs/${id}`),
    enabled: !!id,
  });

  const { data: memberPreview } = useQuery({
    queryKey: ["club-members-preview", id],
    queryFn: () => api.get<PaginatedAthletesResponse>(`/clubs/${id}/members?per_page=4`),
    enabled: !!id,
  });

  const { data: inviteSearch } = useQuery({
    queryKey: ["club-invite-search", inviteQuery],
    queryFn: () =>
      api.get<PaginatedConnectionsResponse>(
        `/athletes/search?query=${encodeURIComponent(inviteQuery)}&per_page=5`,
      ),
    enabled: inviteQuery.length > 1,
  });

  const join = useMutation({
    mutationFn: () => api.post(`/clubs/${id}/join`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["club", id] }),
  });

  const leave = useMutation({
    mutationFn: () => api.delete(`/clubs/${id}/members/${user!.id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["club", id] });
      qc.invalidateQueries({ queryKey: ["club-members-preview", id] });
    },
  });

  const acceptInvite = useMutation({
    mutationFn: () => api.post(`/clubs/${id}/invites/accept`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["club", id] }),
  });

  const denyInvite = useMutation({
    mutationFn: () => api.post(`/clubs/${id}/invites/deny`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["club", id] }),
  });

  const invite = useMutation({
    mutationFn: (athleteId: string) => api.patch(`/clubs/${id}/invite?athlete_id=${athleteId}`),
    onSuccess: () => {
      setInviteMessage("Invite sent.");
      setInviteQuery("");
    },
    onError: () => setInviteMessage("Could not send invite."),
  });

  if (isLoading) return <p className="text-muted">Loading…</p>;
  if (!club) return <p className="text-red-500">Club not found.</p>;

  const isAdmin =
    club.viewer_role === "admin" ||
    club.viewer_role === "owner" ||
    club.admins.includes(user?.id ?? "") ||
    club.creator_id === user?.id;
  const isOwner = club.viewer_role === "owner" || club.creator_id === user?.id;
  const sportTag = club.tags.find((t) => ["running", "run", "cycling", "swim"].includes(t.toLowerCase()));

  const extraMembers = Math.max(0, club.member_count - (memberPreview?.items.length ?? 0));

  return (
    <section>
      <div className="relative -mx-4 sm:-mx-6 lg:-mx-8 mb-0">
        <div
          className="h-40 sm:h-52 w-full bg-border/30 bg-cover bg-center"
          style={
            club.cover_photo_url
              ? { backgroundImage: `url(${club.cover_photo_url})` }
              : { background: "linear-gradient(135deg, var(--color-border) 0%, var(--color-accent)/20 100%)" }
          }
        />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row sm:items-end gap-4 -mt-10 sm:-mt-12 pb-4">
            <ClubAvatar club={club} size="xl" className="border-4 border-global-bg shadow-md" />
            <div className="flex-1 min-w-0 pb-1">
              <h1 className="title text-2xl sm:text-3xl">{club.name}</h1>
              <p className="text-muted text-sm mt-1">
                {club.member_count} {club.member_count === 1 ? "member" : "members"}
                {sportTag && (
                  <>
                    {" "}
                    · <span className="capitalize">{sportTag}</span>
                  </>
                )}
                {club.is_private && " · Private"}
              </p>
            </div>
            <div className="flex flex-wrap gap-2 pb-1">
              {!club.is_member && !club.has_pending_join_request && !club.has_pending_invite && (
                <button
                  type="button"
                  className="btn-primary text-sm"
                  onClick={() => join.mutate()}
                  disabled={join.isPending}
                >
                  {club.is_private ? "Request to Join" : "Join"}
                </button>
              )}
              {club.has_pending_join_request && (
                <button type="button" className="btn-secondary text-sm" disabled>
                  Request Pending
                </button>
              )}
              {club.has_pending_invite && (
                <>
                  <button
                    type="button"
                    className="btn-primary text-sm"
                    onClick={() => acceptInvite.mutate()}
                    disabled={acceptInvite.isPending}
                  >
                    Accept Invite
                  </button>
                  <button
                    type="button"
                    className="btn-secondary text-sm"
                    onClick={() => denyInvite.mutate()}
                    disabled={denyInvite.isPending}
                  >
                    Decline
                  </button>
                </>
              )}
              {club.is_member && !isOwner && (
                <button
                  type="button"
                  className="btn-secondary text-sm"
                  onClick={() => leave.mutate()}
                  disabled={leave.isPending}
                >
                  Leave
                </button>
              )}
              <button type="button" className="btn-secondary text-sm" onClick={() => setReportOpen(true)}>
                Report
              </button>
            </div>
          </div>
        </div>
      </div>

      {club.description && (
        <p className="text-sm leading-relaxed mb-4 max-w-3xl">{club.description}</p>
      )}

      <div className="flex flex-col lg:flex-row gap-8 mt-4">
        <div className="flex-1 min-w-0">
          <nav className="flex flex-wrap gap-1 border-b border-border mb-6">
            {CLUB_TABS.map((tab) => (
              <ClubTab key={tab} active={activeTab === tab} onClick={() => setActiveTab(tab)}>
                {tab === "activity" ? "Recent Activity" : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </ClubTab>
            ))}
          </nav>

          {activeTab === "leaderboard" && id && <ClubLeaderboardTab clubId={id} />}
          {activeTab === "activity" && id && <ClubRecentActivityTab clubId={id} />}
          {activeTab === "members" && id && <ClubMembersTab clubId={id} />}
          {activeTab === "posts" && id && <ClubPostsTab clubId={id} canPost={club.is_member} />}
        </div>

        <aside className="lg:w-72 shrink-0 space-y-6">
          {isAdmin && (
            <div className="card">
              <h2 className="title text-sm mb-3">Invite athletes</h2>
              <input
                className="field-input text-sm mb-2"
                placeholder="Search by name…"
                value={inviteQuery}
                onChange={(e) => setInviteQuery(e.target.value)}
              />
              {inviteMessage && <p className="text-xs text-accent mb-2">{inviteMessage}</p>}
              <ul className="space-y-2">
                {inviteSearch?.items.map((result) => (
                  <li key={result.athlete.id} className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <AthleteAvatar athlete={result.athlete} size="sm" />
                      <span className="text-sm truncate">{athleteName(result.athlete)}</span>
                    </div>
                    <button
                      type="button"
                      className="text-xs text-accent font-semibold shrink-0"
                      onClick={() => invite.mutate(result.athlete.id)}
                    >
                      Invite
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="card">
            <h2 className="title text-sm mb-3">Members</h2>
            <div className="flex flex-wrap gap-1 mb-2">
              {memberPreview?.items.map((a) => (
                <Link key={a.id} to={`/athletes/${a.id}`} title={athleteName(a)}>
                  <AthleteAvatar athlete={a} size="sm" />
                </Link>
              ))}
            </div>
            {extraMembers > 0 && (
              <button
                type="button"
                className="text-xs text-accent font-semibold"
                onClick={() => setActiveTab("members")}
              >
                and {extraMembers} other{extraMembers === 1 ? "" : "s"}
              </button>
            )}
            {club.member_count <= 4 && memberPreview && memberPreview.items.length > 0 && (
              <button
                type="button"
                className="text-xs text-accent font-semibold mt-1"
                onClick={() => setActiveTab("members")}
              >
                View all members
              </button>
            )}
          </div>

          <div className="space-y-2 text-sm">
            {club.is_member && !isOwner && (
              <button
                type="button"
                className="cactus-link block"
                onClick={() => leave.mutate()}
                disabled={leave.isPending}
              >
                Leave club
              </button>
            )}
            {isAdmin && (
              <>
                <Link className="cactus-link block" to={`/clubs/${id}/settings`}>
                  Club settings
                </Link>
                {club.is_private && (
                  <Link className="cactus-link block" to={`/clubs/${id}/join-requests`}>
                    Join requests
                  </Link>
                )}
              </>
            )}
          </div>

          {club.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {club.tags.map((t) => (
                <span key={t} className="text-xs border border-border rounded-none px-2 py-0.5 text-muted">
                  {t}
                </span>
              ))}
            </div>
          )}
        </aside>
      </div>

      <ReportDialog open={reportOpen} targetType="club" targetId={id!} onClose={() => setReportOpen(false)} />
    </section>
  );
}
