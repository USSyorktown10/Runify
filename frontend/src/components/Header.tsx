import { useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Logo } from "@/components/Logo";
import { ThemeToggle } from "@/components/ThemeToggle";
import { SearchModal } from "@/components/SearchModal";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { useAuth } from "@/context/AuthContext";
import { athleteName } from "@/lib/format";

const nav = [
  { to: "/feed", label: "Feed" },
  { to: "/activities", label: "Activities" },
  { to: "/segments", label: "Explore" },
  { to: "/clubs", label: "Clubs" },
  { to: "/routes", label: "Routes" },
];

export function Header() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);

  const { data: unread } = useQuery({
    queryKey: ["notifications-count"],
    queryFn: () => api.get<{ unread_count: number }>("/athlete/notifications/number"),
    refetchInterval: 60000,
  });

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <>
      <header className="sticky top-0 z-50 w-full border-b-2 border-border bg-global-bg">
        <div className="mx-auto flex h-16 w-full max-w-[1600px] items-center gap-4 px-4 sm:px-6 lg:px-8">
          <Logo size="sm" />

          <nav
            aria-label="Main"
            className={`${
              menuOpen
                ? "absolute inset-x-0 top-16 flex flex-col border-b-2 border-border bg-global-bg p-4 lg:static lg:flex lg:flex-row lg:border-0 lg:bg-transparent lg:p-0"
                : "hidden lg:flex"
            } lg:items-center lg:gap-1`}
          >
            {nav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded-none px-2 py-1 text-xs font-semibold ${
                    isActive
                      ? "border-2 border-accent bg-surface text-accent"
                      : "text-muted hover:bg-surface hover:text-global-text"
                  }`
                }
                onClick={() => setMenuOpen(false)}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ms-auto flex items-center gap-1">
            <button
              type="button"
              className="rounded-none p-2 text-muted hover:bg-surface hover:text-accent"
              aria-label="Search"
              onClick={() => setSearchOpen(true)}
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.35-4.35" />
              </svg>
            </button>

            <Link
              to="/notifications"
              className="relative rounded-none p-2 text-muted hover:bg-surface hover:text-accent"
              aria-label="Notifications"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path d="M14.857 17.082a23.848 23.848 0 0 0 5.454-1.31A8.967 8.967 0 0 1 18 9.75V9A6 6 0 0 0 6 9v.75a8.967 8.967 0 0 1-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 0 1-5.714 0m5.714 0a3 3 0 1 1-5.714 0" />
              </svg>
              {(unread?.unread_count ?? 0) > 0 && (
                <span className="absolute end-1 top-1 flex h-4 min-w-4 items-center justify-center rounded-none bg-highlight px-1 text-[10px] font-bold text-global-bg">
                  {unread!.unread_count}
                </span>
              )}
            </Link>

            <ThemeToggle />

            <div className="relative hidden sm:block">
              <button
                type="button"
                className="flex items-center gap-2 rounded-none px-2 py-1 text-xs font-semibold text-muted hover:bg-surface hover:text-accent"
                aria-haspopup="menu"
                aria-expanded={profileOpen}
                onClick={() => setProfileOpen(!profileOpen)}
              >
                {user && <AthleteAvatar athlete={user} size="sm" />}
                {user ? athleteName(user) : "Account"}
              </button>
              {profileOpen && (
                <>
                  <button
                    type="button"
                    className="fixed inset-0 z-40 cursor-default"
                    aria-label="Close menu"
                    onClick={() => setProfileOpen(false)}
                  />
                  <div className="absolute end-0 top-full z-50 mt-1 min-w-44 rounded-none border-2 border-border bg-global-bg py-1">
                    <Link className="block px-4 py-2 text-sm hover:bg-surface hover:text-accent" to="/settings" onClick={() => setProfileOpen(false)}>
                      Settings
                    </Link>
                    {user && (
                      <Link className="block px-4 py-2 text-sm hover:bg-surface hover:text-accent" to={`/athletes/${user.id}`} onClick={() => setProfileOpen(false)}>
                        Profile
                      </Link>
                    )}
                    <button type="button" className="block w-full px-4 py-2 text-start text-sm hover:bg-surface hover:text-accent" onClick={handleLogout}>
                      Log out
                    </button>
                  </div>
                </>
              )}
            </div>

            <button
              type="button"
              className="rounded-none p-2 text-muted hover:bg-surface lg:hidden"
              aria-expanded={menuOpen}
              aria-label="Open main menu"
              onClick={() => setMenuOpen(!menuOpen)}
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                {menuOpen ? <path d="M6 18L18 6M6 6l12 12" /> : <path d="M3.75 9h16.5m-16.5 6.75h16.5" />}
              </svg>
            </button>
          </div>
        </div>
      </header>
      <SearchModal open={searchOpen} onClose={() => setSearchOpen(false)} />
    </>
  );
}
