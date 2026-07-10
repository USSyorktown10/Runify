import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { DetailedRoute } from "@/types/api";

export function EditRoutePage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  useQuery({
    queryKey: ["route", id],
    queryFn: async () => {
      const r = await api.get<DetailedRoute>(`/routes/${id}`);
      setName(r.name);
      setDescription(r.description);
      return r;
    },
    enabled: !!id,
  });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    await api.patch(`/routes/${id}`, { name, description });
    navigate(`/routes/${id}`);
  };

  return (
    <section>
      <h1 className="title mb-8">Edit route</h1>
      <form onSubmit={submit} className="space-y-4 max-w-md">
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Name</label>
          <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Description</label>
          <textarea className="field-input min-h-24" value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>
        <button type="submit" className="btn-primary">
          Save
        </button>
      </form>
    </section>
  );
}
