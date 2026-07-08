import { Identicon } from "@/lib/identicon";
import type { SummaryClub } from "@/types/api";

const sizes = {
  sm: "h-8 w-8",
  md: "h-10 w-10",
  lg: "h-16 w-16",
  xl: "h-20 w-20",
} as const;

export function ClubAvatar({
  club,
  size = "md",
  className = "",
}: {
  club: Pick<SummaryClub, "id" | "name" | "profile_picture_url">;
  size?: keyof typeof sizes;
  className?: string;
}) {
  const sizeClass = sizes[size];

  if (club.profile_picture_url) {
    return (
      <img
        src={club.profile_picture_url}
        alt={club.name}
        className={`${sizeClass} shrink-0 rounded-none object-cover border border-border ${className}`}
      />
    );
  }

  return (
    <Identicon
      seed={`club:${club.id}`}
      title={club.name}
      className={`${sizeClass} shrink-0 rounded-none border border-border ${className}`}
    />
  );
}
