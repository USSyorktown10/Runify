import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { BackButton } from "@/components/BackButton";
import type { IntegrationStatus } from "@/types/api";

const PROVIDERS = ["garmin", "wahoo", "apple_health"];

export function IntegrationsSettingsPage() {
  const qc = useQueryClient();

  const { data: integrations, isLoading } = useQuery({
    queryKey: ["integrations"],
    queryFn: () => api.get<IntegrationStatus[]>("/integrations"),
  });

  const disconnect = useMutation({
    mutationFn: (provider: string) => api.delete(`/integrations/${provider}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["integrations"] }),
  });

  const connect = async (provider: string) => {
    const res = await api.get<{ redirect_url: string }>(`/integrations/${provider}/connect`);
    window.location.href = res.redirect_url;
  };

  return (
    <section>
      <BackButton to="/settings" label="Back to Settings" />
      <h1 className="title mb-8">Integrations</h1>
      {isLoading && <p className="text-muted">Loading…</p>}
      <ul className="space-y-4">
        {PROVIDERS.map((p) => {
          const status = integrations?.find((i) => i.provider === p);
          return (
            <li key={p} className="card flex justify-between items-center ">
              <span>{p.replace("_", " ")}</span>
              {status?.is_connected ? (
                <button type="button" className="btn-secondary text-sm" onClick={() => disconnect.mutate(p)}>
                  Disconnect
                </button>
              ) : (
                <button type="button" className="btn-primary text-sm" onClick={() => connect(p)}>
                  Connect
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
