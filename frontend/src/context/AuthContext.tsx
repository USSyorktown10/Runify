import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, setOnUnauthorized, setTokenGetter } from "@/api/client";
import { initTheme } from "@/lib/theme";
import type { MeAthlete } from "@/types/api";

const TOKEN_KEY = "runify_session";

interface AuthContextValue {
  token: string | null;
  user: MeAthlete | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<string | null>;
  signup: (data: {
    username: string;
    email: string;
    password: string;
    metadata?: Record<string, unknown>;
  }) => Promise<string | null>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => sessionStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState<MeAthlete | null>(null);
  const [loading, setLoading] = useState(true);

  const clearAuth = useCallback(() => {
    sessionStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    if (!sessionStorage.getItem(TOKEN_KEY)) {
      setUser(null);
      return;
    }
    try {
      const me = await api.get<MeAthlete>("/athlete/me");
      setUser(me);
      initTheme();
    } catch {
      clearAuth();
    }
  }, [clearAuth]);

  useEffect(() => {
    setTokenGetter(() => sessionStorage.getItem(TOKEN_KEY));
    setOnUnauthorized(clearAuth);
    initTheme();
    refreshUser().finally(() => setLoading(false));
  }, [clearAuth, refreshUser]);

  const login = useCallback(async (username: string, password: string) => {
    const res = await api.post<{ session_token?: string; error_message?: string }>(
      "/authentication/login",
      { username, password },
      false,
    );
    if (!res.session_token) return res.error_message ?? "Login failed";
    sessionStorage.setItem(TOKEN_KEY, res.session_token);
    setToken(res.session_token);
    await refreshUser();
    return null;
  }, [refreshUser]);

  const signup = useCallback(
    async (data: {
      username: string;
      email: string;
      password: string;
      metadata?: Record<string, unknown>;
    }) => {
      const res = await api.post<{ success: boolean; error_message?: string }>(
        "/authentication/signup",
        data,
        false,
      );
      if (!res.success) return res.error_message ?? "Signup failed";
      return null;
    },
    [],
  );

  const logout = useCallback(async () => {
    const t = sessionStorage.getItem(TOKEN_KEY);
    if (t) {
      try {
        await api.post("/authentication/logout", { session_token: t }, false);
      } catch {
        /* ignore */
      }
    }
    clearAuth();
  }, [clearAuth]);

  const value = useMemo(
    () => ({ token, user, loading, login, signup, logout, refreshUser }),
    [token, user, loading, login, signup, logout, refreshUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
