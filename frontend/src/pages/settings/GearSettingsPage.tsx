import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { EmptyState } from "@/components/EmptyState";
import { BackButton } from "@/components/BackButton";
import type { Gear } from "@/types/api";

export function GearSettingsPage() {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [deleteId, setDeleteId] = useState<string | null>(null);

  const { data: gear, isLoading } = useQuery({
    queryKey: ["gear"],
    queryFn: () => api.get<Gear[]>("/gear"),
  });

  const create = useMutation({
    mutationFn: () =>
      api.post("/gear", {
        name,
        brand_name: "",
        model_name: "",
        initial_date: new Date().toISOString().slice(0, 10),
        max_mileage: 500000,
      }),
    onSuccess: () => {
      setName("");
      qc.invalidateQueries({ queryKey: ["gear"] });
    },
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/gear/${id}`),
    onSuccess: () => {
      setDeleteId(null);
      qc.invalidateQueries({ queryKey: ["gear"] });
    },
  });

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Gear locker</h1>
      <form
        className="flex gap-2 mb-8"
        onSubmit={(e) => {
          e.preventDefault();
          if (name) create.mutate();
        }}
      >
        <input className="field-input flex-1" placeholder="New gear name" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="btn-primary">
          Add
        </button>
      </form>
      {isLoading && <p className="text-muted">Loading…</p>}
      {!isLoading && gear?.length === 0 && <EmptyState title="No gear yet" />}
      <ul className="space-y-2">
        {gear?.map((g) => (
          <li key={g.id} className="card flex justify-between items-center">
            <Link className="cactus-link" to={`/settings/gear/${g.id}`}>
              {g.name}
            </Link>
            <button type="button" className="text-red-500 text-sm" onClick={() => setDeleteId(g.id)}>
              Delete
            </button>
          </li>
        ))}
      </ul>
      <ConfirmDialog
        open={!!deleteId}
        title="Delete gear"
        message="Remove this item from your locker?"
        danger
        confirmLabel="Delete"
        onConfirm={() => deleteId && remove.mutate(deleteId)}
        onCancel={() => setDeleteId(null)}
      />
    </section>
  );
}
