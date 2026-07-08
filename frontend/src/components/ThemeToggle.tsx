import { useEffect, useState } from "react";
import { api } from "@/api/client";
import { useAuth } from "@/context/AuthContext";
import { applyTheme, getEffectiveTheme, toggleTheme, type Theme } from "@/lib/theme";

export function ThemeToggle() {
  const { token } = useAuth();
  const [theme, setTheme] = useState<Theme>(() => getEffectiveTheme());

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const handleToggle = async () => {
    const next = toggleTheme(theme);
    setTheme(next);
    if (token) {
      try {
        await api.patch("/preferences", { theme: next });
      } catch {
        /* local only */
      }
    }
  };

  return (
    <button
      type="button"
      role="switch"
      aria-checked={theme === "dark"}
      className="hover:text-accent relative h-9 w-9 cursor-pointer rounded-none p-2 hover:bg-surface"
      onClick={handleToggle}
    >
      <span className="sr-only">{theme === "dark" ? "Dark theme" : "Light theme"}</span>
      {theme === "dark" ? (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 0 7.92 12.446a9 9 0 1 1 -8.313 -12.454z" />
        </svg>
      ) : (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <circle cx="12" cy="12" r="5" />
          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
        </svg>
      )}
    </button>
  );
}
