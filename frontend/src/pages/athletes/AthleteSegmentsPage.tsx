import { Navigate, useParams } from "react-router-dom";

export function AthleteSegmentsPage() {
  const { id } = useParams<{ id: string }>();
  return <Navigate to={`/athletes/${id}?tab=segments`} replace />;
}
