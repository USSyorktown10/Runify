import type { DetailedActivity, SummaryActivity } from "@/types/api";

export function canViewBiometrics(
  activity: Pick<SummaryActivity | DetailedActivity, "biometrics_visibility">,
  isOwner: boolean,
  isFollowing: boolean,
): boolean {
  if (isOwner) return true;
  const v = activity.biometrics_visibility;
  if (v === "public") return true;
  if (v === "followers" && isFollowing) return true;
  return false;
}
