import { athleteName } from "@/lib/format";
import { avatarSeed, type AvatarAthlete } from "@/lib/avatar";
import { Identicon } from "@/lib/identicon";

const sizes = {
  sm: "h-8 w-8",
  md: "h-10 w-10",
  lg: "h-16 w-16",
  xl: "h-20 w-20",
} as const;

export function AthleteAvatar({
  athlete,
  size = "md",
  className = "",
}: {
  athlete: AvatarAthlete;
  size?: keyof typeof sizes;
  className?: string;
}) {
  const label = athleteName(athlete);
  const sizeClass = sizes[size];

  if (athlete.profile_picture_url) {
    return (
      <img
        src={athlete.profile_picture_url}
        alt={label}
        className={`${sizeClass} shrink-0 rounded-none object-cover ${className}`}
      />
    );
  }

  return (
    <Identicon
      seed={avatarSeed(athlete)}
      title={label}
      className={`${sizeClass} shrink-0 rounded-none ${className}`}
    />
  );
}
