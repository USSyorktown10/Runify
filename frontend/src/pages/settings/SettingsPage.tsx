import { Link } from "react-router-dom";

export function SettingsPage() {
  const links = [
    { to: "/settings/profile", label: "Profile" },
    { to: "/settings/preferences", label: "Preferences" },
    { to: "/settings/stats", label: "Training stats" },
    { to: "/settings/gear", label: "Gear locker" },
    { to: "/settings/integrations", label: "Integrations" },
    { to: "/settings/sessions", label: "Active sessions" },
    { to: "/settings/privacy", label: "Privacy & blocks" },
    { to: "/settings/follow-requests", label: "Follow requests" },
  ];

  return (
    <section>
      <div className="page-header">
        <h1 className="title text-3xl">Settings</h1>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {links.map((l) => (
          <Link
            key={l.to}
            className="card p-3 font-semibold hover:border-accent"
            to={l.to}
          >
            {l.label}
          </Link>
        ))}
      </div>
    </section>
  );
}
