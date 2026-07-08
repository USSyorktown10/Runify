import { Navigate, useParams } from "react-router-dom";

export function AthleteFollowersPage() {
  const { id } = useParams<{ id: string }>();
  return <Navigate to={`/athletes/${id}?tab=followers`} replace />;
}

export function AthleteFollowingPage() {
  const { id } = useParams<{ id: string }>();
  return <Navigate to={`/athletes/${id}?tab=following`} replace />;
}
