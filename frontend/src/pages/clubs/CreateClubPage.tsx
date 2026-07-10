import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";

export function CreateClubPage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isPrivate, setIsPrivate] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await api.post<{ success: boolean; club?: { id: string } }>("/clubs", {
        name,
        description,
        is_private: isPrivate,
        tags: ["running"],
      });
      if (res.club?.id) navigate(`/clubs/${res.club.id}`);
      else setError("Create failed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Create failed");
    }
  };

  return (
    <section>
      <h1 className="title mb-8">Create club</h1>
      <form onSubmit={submit} className="space-y-4 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Description</label>
          <textarea className="field-input min-h-24" value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={isPrivate} onChange={(e) => setIsPrivate(e.target.checked)} />
          Private club
        </label>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary">
          Create
        </button>
      </form>
    </section>
  );
}
