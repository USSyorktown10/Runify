import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import { usePreferences } from "@/hooks/usePreferences";
import { BackButton } from "@/components/BackButton";

export function PreferencesSettingsPage() {
  const { data: prefs, refetch } = usePreferences();

  const save = useMutation({
    mutationFn: (body: Record<string, unknown>) => api.patch("/preferences", body),
    onSuccess: () => refetch(),
  });

  if (!prefs) return <p className="text-muted">Loading…</p>;

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Preferences</h1>
      <div className="space-y-6 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Units</label>
          <select
            className="field-input"
            value={prefs.measurement_system}
            onChange={(e) => save.mutate({ measurement_system: e.target.value })}
          >
            <option value="metric">Metric</option>
            <option value="imperial">Imperial</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Theme</label>
          <select
            className="field-input"
            value={prefs.theme}
            onChange={(e) => save.mutate({ theme: e.target.value })}
          >
            <option value="system">System</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
        <fieldset>
          <legend className="text-xs text-muted font-semibold uppercase mb-2">Privacy defaults</legend>
          {(["profile_visibility", "activity_visibility", "biometrics_visibility"] as const).map((key) => (
            <div key={key} className="mb-3">
              <label className="block text-sm mb-1 capitalize">{key.replace(/_/g, " ")}</label>
              <select
                className="field-input"
                value={prefs.privacy_settings[key]}
                onChange={(e) =>
                  save.mutate({
                    privacy_settings: { ...prefs.privacy_settings, [key]: e.target.value },
                  })
                }
              >
                <option value="public">Public</option>
                <option value="followers">Followers</option>
                <option value="private">Private</option>
              </select>
            </div>
          ))}
        </fieldset>
        <fieldset>
          <legend className="text-xs text-muted font-semibold uppercase mb-2">Email notifications</legend>
          {(["comments", "likes", "follow_requests", "club_invites"] as const).map((key) => (
            <label key={key} className="flex items-center gap-2 mb-2 capitalize">
              <input
                type="checkbox"
                checked={prefs.email_notifications[key]}
                onChange={(e) =>
                  save.mutate({
                    email_notifications: { ...prefs.email_notifications, [key]: e.target.checked },
                  })
                }
              />
              {key.replace(/_/g, " ")}
            </label>
          ))}
        </fieldset>
      </div>
    </section>
  );
}
