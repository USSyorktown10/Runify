import { useParams, Navigate } from "react-router-dom";

export function AthleteFeedPage() {
  const { id } = useParams<{ id: string }>();
  return <Navigate to={`/athletes/${id}`} replace />;
}
