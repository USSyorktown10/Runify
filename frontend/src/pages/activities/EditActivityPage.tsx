import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { DetailedActivity } from "@/types/api";

export function EditActivityPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useQuery({
    queryKey: ["activity", id],
    queryFn: async () => {
      const a = await api.get<DetailedActivity>(`/activities/${id}`);
      if (!initialized) {
        setName(a.name);
        setDescription(a.description);
        setInitialized(true);
      }
      return a;
    },
    enabled: !!id,
  });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await api.patch<{ success: boolean; error_message?: string }>(`/activities/${id}`, {
        name,
        description,
      });
      if (!res.success) {
        setError(res.error_message ?? "Update failed");
        return;
      }
      navigate(`/activities/${id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Update failed");
    }
  };

  return (
    <section>
      <h1 className="title mb-8">Edit activity</h1>
      <form onSubmit={submit} className="space-y-4 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Description</label>
          <textarea className="field-input min-h-24" value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary">
          Save
        </button>
      </form>
    </section>
  );
}
