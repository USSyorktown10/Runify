import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { RunPhraseMarqueeBackground } from "@/components/RunPhraseHero";

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const err = await login(username, password);
    setLoading(false);
    if (err) {
      setError(err);
      return;
    }
    navigate(params.get("redirect") ?? "/feed");
  };

  return (
    <div className="relative flex min-h-screen w-full flex-col justify-center items-center overflow-hidden px-4 py-12">
      <RunPhraseMarqueeBackground />

      <div className="absolute inset-y-0 start-0 z-[1] w-1 bg-accent" aria-hidden="true" />

      <div className="relative z-10 w-full max-w-md border-2 border-border bg-global-bg p-6 sm:p-8">
        <div className="mb-6 flex flex-col items-center text-center">
          <img src="/logo.svg" alt="" className="mb-3 h-12 w-12 border border-border" />
          <h1 className="title">Time To Run</h1>
        </div>


        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="label mb-1 block">Username</label>
            <input className="field-input" value={username} onChange={(e) => setUsername(e.target.value)} required />
          </div>
          <div>
            <label className="label mb-1 block">Password</label>
            <input
              className="field-input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && <p className="text-red-500 font-mono text-xs">{error}</p>}
          <button type="submit" className="btn-primary w-full" disabled={loading}>
            {loading ? "Logging in…" : "Log in"}
          </button>
        </form>

        <p className="mt-6 text-muted text-center text-xs">
          <Link className="cactus-link" to="/forgot-password">
            Forgot password?
          </Link>
          {" · "}
          <Link className="cactus-link" to="/signup">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
