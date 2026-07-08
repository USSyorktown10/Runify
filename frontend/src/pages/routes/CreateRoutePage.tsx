import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";

export function CreateRoutePage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [activityId, setActivityId] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await api.post<{ success: boolean; route?: { id: string } }>("/routes", {
        name,
        activity_type: "run",
        activity_id: activityId || undefined,
        is_private: false,
      });
      if (res.route?.id) navigate(`/routes/${res.route.id}`);
      else setError("Create failed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Create failed");
    }
  };

  return (
    <section>
      <h1 className="title mb-8">New route</h1>
      <form onSubmit={submit} className="space-y-4 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">From activity ID (optional)</label>
          <input className="field-input" value={activityId} onChange={(e) => setActivityId(e.target.value)} />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary">
          Create
        </button>
      </form>
    </section>
  );
}
