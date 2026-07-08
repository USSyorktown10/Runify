import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import { BackButton } from "@/components/BackButton";

export function StatsSettingsPage() {
  const [ftp, setFtp] = useState("");
  const [pace, setPace] = useState("");
  const [saved, setSaved] = useState(false);

  const save = useMutation({
    mutationFn: () =>
      api.patch("/athlete/stats", {
        current_ftp: ftp ? parseInt(ftp, 10) : undefined,
        threshold_pace: pace ? parseFloat(pace) : undefined,
      }),
    onSuccess: () => setSaved(true),
  });

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Training stats</h1>
      <form
        className="space-y-4 max-w-md"
        onSubmit={(e) => {
          e.preventDefault();
          save.mutate();
        }}
      >
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">FTP (watts)</label>
          <input className="field-input" type="number" value={ftp} onChange={(e) => setFtp(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Threshold pace (m/s)</label>
          <input className="field-input" type="number" step="0.01" value={pace} onChange={(e) => setPace(e.target.value)} />
        </div>
        {saved && <p className="text-accent">Saved.</p>}
        <button type="submit" className="btn-primary">
          Save
        </button>
      </form>
    </section>
  );
}
