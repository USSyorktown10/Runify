import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api/client";
import type { DetailedClub } from "@/types/api";

export function ClubSettingsPage() {
  const { id } = useParams<{ id: string }>();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saved, setSaved] = useState(false);

  useQuery({
    queryKey: ["club", id],
    queryFn: async () => {
      const c = await api.get<DetailedClub>(`/clubs/${id}`);
      setName(c.name);
      setDescription(c.description);
      return c;
    },
    enabled: !!id,
  });

  const save = useMutation({
    mutationFn: () => api.patch(`/clubs/${id}/preferences`, { name, description }),
    onSuccess: () => setSaved(true),
  });

  return (
    <section>
      <h1 className="title mb-8">Club settings</h1>
      <Link className="cactus-link text-sm mb-4 inline-block" to={`/clubs/${id}`}>
        ← Back to club
      </Link>
      <form
        className="space-y-4 max-w-md mt-4"
        onSubmit={(e) => {
          e.preventDefault();
          save.mutate();
        }}
      >
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Description</label>
          <textarea className="field-input min-h-24" value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>
        {saved && <p className="text-accent">Saved.</p>}
        <button type="submit" className="btn-primary">
          Save
        </button>
      </form>
      <section className="mt-12">
        <h2 className="title text-lg mb-4">Join requests</h2>
        <p className="text-muted text-sm mb-2">Review athletes waiting to join this club.</p>
        <Link className="cactus-link text-sm inline-block" to={`/clubs/${id}/join-requests`}>
          View join requests →
        </Link>
      </section>
    </section>
  );
}
