import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { useAuth } from "@/context/AuthContext";
import { BackButton } from "@/components/BackButton";

export function ProfileSettingsPage() {
  const { user, refreshUser } = useAuth();
  const [firstName, setFirstName] = useState(user?.first_name ?? "");
  const [lastName, setLastName] = useState(user?.last_name ?? "");
  const [city, setCity] = useState(user?.city ?? "");
  const [saved, setSaved] = useState(false);

  const save = useMutation({
    mutationFn: () =>
      api.patch("/athlete/profile", {
        first_name: firstName,
        last_name: lastName,
        city,
      }),
    onSuccess: async () => {
      await refreshUser();
      setSaved(true);
    },
  });

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Profile</h1>
      {user && (
        <div className="mb-6 flex items-center gap-4">
          <AthleteAvatar athlete={user} size="lg" />
          <p className="font-semibold text-global-text">@{user.username}</p>
        </div>
      )}
      <form
        className="space-y-4 max-w-md"
        onSubmit={(e) => {
          e.preventDefault();
          save.mutate();
        }}
      >
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">First name</label>
          <input className="field-input" value={firstName} onChange={(e) => setFirstName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Last name</label>
          <input className="field-input" value={lastName} onChange={(e) => setLastName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">City</label>
          <input className="field-input" value={city} onChange={(e) => setCity(e.target.value)} />
        </div>
        {saved && <p className="text-accent">Saved.</p>}
        <button type="submit" className="btn-primary" disabled={save.isPending}>
          Save
        </button>
      </form>
    </section>
  );
}
