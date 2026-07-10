import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";

export function CreateActivityPage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [startDate, setStartDate] = useState(new Date().toISOString().slice(0, 16));
  const [distance, setDistance] = useState("");
  const [elapsedTime, setElapsedTime] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await api.post<{ success: boolean; id?: string; error_message?: string }>("/activities", {
        name,
        start_date: new Date(startDate).toISOString(),
        distance: parseFloat(distance) * 1000,
        elapsed_time: parseInt(elapsedTime, 10) * 60,
        description,
        activity_type: "run",
      });
      if (!res.success || !res.id) {
        setError(res.error_message ?? "Create failed");
        return;
      }
      navigate(`/activities/${res.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Create failed");
    }
  };

  return (
    <section>
      <h1 className="title mb-8">New activity</h1>
      <form onSubmit={submit} className="space-y-4 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Start</label>
          <input className="field-input" type="datetime-local" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Distance (km)</label>
          <input className="field-input" type="number" step="0.01" value={distance} onChange={(e) => setDistance(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Duration (minutes)</label>
          <input className="field-input" type="number" value={elapsedTime} onChange={(e) => setElapsedTime(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Description</label>
          <textarea className="field-input min-h-24" value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary">
          Create
        </button>
      </form>
    </section>
  );
}
