export type Theme = "light" | "dark";

const STORAGE_KEY = "theme";

export function getStoredTheme(): Theme | null {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return null;
}

export function getSystemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function getEffectiveTheme(pref?: string | null): Theme {
  if (pref === "light" || pref === "dark") return pref;
  if (pref === "system" || !pref) {
    return getStoredTheme() ?? getSystemTheme();
  }
  return getStoredTheme() ?? getSystemTheme();
}

export function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(STORAGE_KEY, theme);
}

export function initTheme(pref?: string | null) {
  applyTheme(getEffectiveTheme(pref));
}

export function toggleTheme(current: Theme): Theme {
  const next = current === "light" ? "dark" : "light";
  applyTheme(next);
  return next;
}
