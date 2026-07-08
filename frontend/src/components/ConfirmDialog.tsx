export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  onConfirm,
  onCancel,
  danger,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;
}) {
  if (!open) return null;
  return (
    <div className="modal-overlay fixed inset-0 flex items-center justify-center bg-global-bg/90 p-4">
      <div className="card max-w-md w-full border-2 bg-global-bg">
        <h2 className="title text-lg mb-2">{title}</h2>
        <p className="text-muted mb-6">{message}</p>
        <div className="flex gap-2 justify-end">
          <button type="button" className="btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className={danger ? "btn-primary bg-red-600" : "btn-primary"}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
