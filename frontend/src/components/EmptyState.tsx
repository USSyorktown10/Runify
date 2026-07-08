export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="card text-center py-12">
      <h2 className="title text-lg mb-2">{title}</h2>
      {description && <p className="text-muted mb-4">{description}</p>}
      {action}
    </div>
  );
}
