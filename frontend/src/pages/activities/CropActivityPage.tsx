import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "@/api/client";

export function CropActivityPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [startIndex, setStartIndex] = useState(0);
  const [endIndex, setEndIndex] = useState(100);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await api.post<{ success: boolean }>(`/activities/${id}/crop`, {
        start_index: startIndex,
        end_index: endIndex,
      });
      if (res.success) navigate(`/activities/${id}`);
      else setError("Crop failed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Crop failed");
    }
  };

  return (
    <section>
      <h1 className="title mb-8">Crop activity</h1>
      <p className="text-muted mb-6">Trim data points from the start or end of the activity stream.</p>
      <form onSubmit={submit} className="space-y-4 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">Start index</label>
          <input className="field-input" type="number" value={startIndex} onChange={(e) => setStartIndex(+e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">End index</label>
          <input className="field-input" type="number" value={endIndex} onChange={(e) => setEndIndex(+e.target.value)} />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary">
          Crop
        </button>
      </form>
    </section>
  );
}
