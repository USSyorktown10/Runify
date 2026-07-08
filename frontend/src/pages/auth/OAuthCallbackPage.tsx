import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";
import { useAuth } from "@/context/AuthContext";

export function OAuthCallbackPage({ provider }: { provider: "google" | "apple" }) {
  const navigate = useNavigate();
  const { refreshUser } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const hash = window.location.hash.slice(1);
    const params = new URLSearchParams(hash || window.location.search.slice(1));
    const oauthToken = params.get("access_token") ?? params.get("id_token") ?? params.get("code");

    if (!oauthToken) {
      setError("No OAuth token received.");
      return;
    }

    api
      .post<{ session_token?: string; success?: boolean; error_message?: string }>(
        `/authentication/sso/${provider}`,
        {
          oauth_token: oauthToken,
          client_metadata: {
            user_agent: navigator.userAgent,
            browser_name: "",
            browser_version: "",
            os_name: "",
          },
        },
        false,
      )
      .then(async (res) => {
        if (res.session_token) {
          sessionStorage.setItem("runify_session", res.session_token);
          await refreshUser();
          navigate("/feed");
        } else {
          setError(res.error_message ?? "SSO failed");
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "SSO failed"));
  }, [provider, navigate, refreshUser]);

  if (error) return <p className="text-red-500 text-center py-20">{error}</p>;
  return <p className="text-muted text-center py-20">Completing sign in…</p>;
}
