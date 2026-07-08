import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { AthletePreferences } from "@/types/api";

export function usePreferences() {
  return useQuery({
    queryKey: ["preferences"],
    queryFn: () => api.get<AthletePreferences>("/preferences"),
    retry: false,
  });
}
