import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { BackButton } from "@/components/BackButton";
import type { Gear } from "@/types/api";

export function GearDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [name, setName] = useState("");

  const { data: gear } = useQuery({
    queryKey: ["gear", id],
    queryFn: async () => {
      const g = await api.get<Gear>(`/gear/${id}`);
      setName(g.name);
      return g;
    },
    enabled: !!id,
  });

  const save = useMutation({
    mutationFn: () => api.patch(`/gear/${id}`, { name }),
    onSuccess: () => navigate("/settings/gear"),
  });

  if (!gear) return <p className="text-muted">Loading…</p>;

  return (
    <section>
      <BackButton to="/settings/gear" label="Back to Gear Locker" />
      <h1 className="title mb-8">Edit gear</h1>
      <form
        className="space-y-4 max-w-md"
        onSubmit={(e) => {
          e.preventDefault();
          save.mutate();
        }}
      >
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <p className="text-muted text-sm">
          {gear.brand_name} {gear.model_name} · {gear.gear_type}
        </p>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={gear.is_retired}
            onChange={(e) => api.patch(`/gear/${id}`, { is_retired: e.target.checked })}
          />
          Retired
        </label>
        <button type="submit" className="btn-primary">
          Save
        </button>
      </form>
    </section>
  );
}
