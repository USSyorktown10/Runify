import type { LeaderboardEntry, SummaryAthlete } from "@/types/api";

export type AvatarAthlete = Pick<SummaryAthlete, "first_name" | "last_name" | "profile_picture_url"> & {
  id?: string;
};

export function avatarSeed(athlete: AvatarAthlete): string {
  return athlete.id ?? `${athlete.first_name}|${athlete.last_name}`.trim();
}

export function athleteFromLeaderboard(entry: LeaderboardEntry): AvatarAthlete & { id: string } {
  const parts = entry.athlete_name.trim().split(/\s+/);
  return {
    id: entry.athlete_id,
    first_name: parts[0] ?? "",
    last_name: parts.slice(1).join(" "),
    profile_picture_url: entry.athlete_profile_picture_url,
  };
}
