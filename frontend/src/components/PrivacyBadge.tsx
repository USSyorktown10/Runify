export function PrivacyBadge({ visibility }: { visibility: string }) {
  if (visibility === "public") return null;
  return (
    <span className="text-xs text-muted border border-border rounded-none px-1.5 py-0.5">
      {visibility === "private" ? "Private" : "Followers only"}
    </span>
  );
}

export function PrivacyVeil({ message = "Biometrics hidden" }: { message?: string }) {
  return (
    <div className="card text-center text-muted py-8">
      <span aria-hidden="true" className="text-2xl">
        🔒
      </span>
      <p className="mt-2">{message}</p>
    </div>
  );
}
