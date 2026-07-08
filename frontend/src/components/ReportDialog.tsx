import { useState } from "react";
import { api } from "@/api/client";

export function ReportDialog({
  open,
  targetType,
  targetId,
  onClose,
}: {
  open: boolean;
  targetType: "activity" | "athlete" | "club";
  targetId: string;
  onClose: () => void;
}) {
  const [reason, setReason] = useState("");
  const [details, setDetails] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  if (!open) return null;

  const path =
    targetType === "activity"
      ? `/activities/${targetId}/report`
      : targetType === "athlete"
        ? `/athletes/${targetId}/report`
        : `/clubs/${targetId}/report`;

  const submit = async () => {
    setError(null);
    try {
      const params = new URLSearchParams({ reason, details });
      const res = await api.post<{ success: boolean; error_message?: string }>(
        `${path}?${params.toString()}`,
      );
      if (!res.success) {
        setError(res.error_message ?? "Report failed");
        return;
      }
      setDone(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Report failed");
    }
  };

  return (
    <div className="modal-overlay fixed inset-0 flex items-center justify-center bg-global-bg/90 p-4">
      <div className="card max-w-md w-full border-2 bg-global-bg">
        <h2 className="title text-lg mb-4">Report content</h2>
        {done ? (
          <p className="text-accent mb-4">Thank you. Your report has been submitted.</p>
        ) : (
          <>
            <label className="block text-xs text-muted font-semibold uppercase mb-1">Reason</label>
            <input className="field-input mb-3" value={reason} onChange={(e) => setReason(e.target.value)} />
            <label className="block text-xs text-muted font-semibold uppercase mb-1">Details</label>
            <textarea
              className="field-input mb-3 min-h-24"
              value={details}
              onChange={(e) => setDetails(e.target.value)}
            />
            {error && <p className="text-red-500 mb-3">{error}</p>}
          </>
        )}
        <div className="flex gap-2 justify-end">
          <button type="button" className="btn-secondary" onClick={onClose}>
            Close
          </button>
          {!done && (
            <button type="button" className="btn-primary" onClick={submit} disabled={!reason}>
              Submit
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
