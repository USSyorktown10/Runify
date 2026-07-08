import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <section className="text-center py-20">
      <h1 className="title text-3xl mb-4">404</h1>
      <p className="text-muted mb-8">Page not found.</p>
      <Link className="btn-primary" to="/">
        Go home
      </Link>
    </section>
  );
}
