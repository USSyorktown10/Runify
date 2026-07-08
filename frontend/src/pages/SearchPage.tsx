import { Link } from "react-router-dom";

export function SearchPage() {
  return (
    <section>
      <h1 className="title mb-8">Search</h1>
      <p className="text-muted mb-4">Use the search icon in the header, or browse:</p>
      <ul className="space-y-2">
        <li>
          <Link className="cactus-link" to="/segments">
            Segments
          </Link>
        </li>
        <li>
          <Link className="cactus-link" to="/clubs">
            Clubs
          </Link>
        </li>
      </ul>
    </section>
  );
}
