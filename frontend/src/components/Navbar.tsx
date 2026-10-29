import React from 'react';
import '@/styles/navbar.css';
import { Logo } from '../components/Logo';

//Placeholders
const user = { id: "1", name: "Jamal" };
const athleteName = (u) => u?.name || "Account";
const unread = { unread_count: 3 };

const nav = [
  { to: "/feed", label: "Feed" },
  { to: "/activities", label: "Activities" },
  { to: "/segments", label: "Explore" },
  { to: "/clubs", label: "Clubs" },
  { to: "/routes", label: "Routes" },
];

export function Navbar() {
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [profileOpen, setProfileOpen] = React.useState(false);

  const handleLogout = () => {
    console.log("Logged out");
  };

  return (
    <header className="header-bar">
      <div className="header-inner">
        <Logo size="sm" />

        <nav
          aria-label="Main"
          className={`nav-links ${menuOpen ? "menu-open" : ""}`}
        >
          {nav.map((item) => {
            const isActive = typeof window !== 'undefined' && window.location.pathname === item.to;
            return (
              <a
                key={item.to}
                href={item.to}
                className={`nav-link ${isActive ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  setMenuOpen(false);
                }}
              >
                {item.label}
              </a>
            );
          })}
        </nav>

        <div className="actions-group">
          <button
            type="button"
            className="icon-button"
            aria-label="Search"
            onClick={() => console.log("Open search placeholder")}
          >
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ width: '1.25rem', height: '1.25rem' }}>
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          </button>

          <a
            href="/notifications"
            className="icon-button"
            style={{ position: 'relative', display: 'inline-flex' }}
            aria-label="Notifications"
            onClick={(e) => e.preventDefault()}
          >
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} style={{ width: '1.25rem', height: '1.25rem' }}>
              <path d="M14.857 17.082a23.848 23.848 0 0 0 5.454-1.31A8.967 8.967 0 0 1 18 9.75V9A6 6 0 0 0 6 9v.75a8.967 8.967 0 0 1-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 0 1-5.714 0m5.714 0a3 3 0 1 1-5.714 0" />
            </svg>
            {(unread?.unread_count ?? 0) > 0 && (
              <span className="badge">
                {unread.unread_count}
              </span>
            )}
          </a>

          <button 
            type="button" 
            className="icon-button" 
            aria-label="Toggle theme"
            onClick={() => console.log("Toggle theme placeholder")}
          >
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} style={{ width: '1.25rem', height: '1.25rem' }}>
              <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M14 12a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </button>

          <div className="profile-container">
            <button
              type="button"
              className="profile-button"
              aria-haspopup="menu"
              aria-expanded={profileOpen}
              onClick={() => setProfileOpen(!profileOpen)}
            >
              {user && (
                <div className="avatar-placeholder" style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#ccc', display: 'inline-block' }} />
              )}
              {user ? athleteName(user) : "Account"}
            </button>
            {profileOpen && (
              <>
                <button
                  type="button"
                  className="backdrop-fixed"
                  aria-label="Close menu"
                  onClick={() => setProfileOpen(false)}
                />
                <div className="dropdown-menu">
                  <a className="dropdown-item" href="/settings" onClick={(e) => { e.preventDefault(); setProfileOpen(false); }}>
                    Settings
                  </a>
                  {user && (
                    <a className="dropdown-item" href={`/athletes/${user.id}`} onClick={(e) => { e.preventDefault(); setProfileOpen(false); }}>
                      Profile
                    </a>
                  )}
                  <button type="button" className="dropdown-item" onClick={handleLogout}>
                    Log out
                  </button>
                </div>
              </>
            )}
          </div>

          <button
            type="button"
            className="mobile-toggle"
            aria-expanded={menuOpen}
            aria-label="Open main menu"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} style={{ width: '1.5rem', height: '1.5rem' }}>
              {menuOpen ? <path d="M6 18L18 6M6 6l12 12" /> : <path d="M3.75 9h16.5m-16.5 6.75h16.5" />}
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}
