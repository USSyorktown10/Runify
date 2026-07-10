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
      id="theme-toggle"
      type="button"
      className="text-zinc-500 hover:text-accent transition-colors cursor-pointer font-mono text-xs flex items-center gap-0.5"
      onClick={handleToggle}
      aria-label="Toggle Theme"
    >
      theme/
      <i
        id="theme-toggle-icon"
        className={`bi bi-${theme === "dark" ? "sun" : "moon"} inline-block align-middle pointer-events-none`}
      />
    </button>
  );
}
