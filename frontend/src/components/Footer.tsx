import { Link } from "react-router-dom";

export function Footer() {
  return (
    <footer className="mt-auto w-full border-t-2 border-border bg-surface">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col items-center justify-between gap-4 px-4 py-6 text-center sm:flex-row sm:px-6 sm:text-start lg:px-8">
        <div className="flex items-center gap-3">
          <img src="/logo.svg" alt="" className="h-8 w-8" />
          <span className="text-lg font-mono font-semibold">Runify</span>
        </div>
        <p className="text-muted text-xs font-semibold">© Runify {new Date().getFullYear()}</p>
        <nav aria-label="Footer" className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-xs font-semibold">
          <Link className="text-muted hover:text-accent" to="/feed">
            Feed
          </Link>
          <Link className="text-muted hover:text-accent" to="/activities">
            Activities
          </Link>
          <Link className="text-muted hover:text-accent" to="/settings">
            Settings
          </Link>
        </nav>
      </div>
    </footer>
  );
}
