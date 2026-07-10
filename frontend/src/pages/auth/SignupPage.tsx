import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";

export function SignupPage() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const { signup } = useAuth();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const err = await signup({ username, email, password });
    setLoading(false);
    if (err) {
      setError(err);
      return;
    }
    setSuccess(true);
  };

  if (success) {
    return (
      <>
        <h1 className="title mb-4 text-2xl">Check your email</h1>
        <p className="text-muted mb-6">We sent a verification link to {email}.</p>
        <Link className="btn-primary" to="/login">
          Go to login
        </Link>
      </>
    );
  }

  return (
    <>
      <h1 className="title mb-8 text-2xl">Sign up</h1>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Username</label>
          <input className="field-input" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Email</label>
          <input className="field-input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Password</label>
          <input
            className="field-input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary w-full" disabled={loading}>
          {loading ? "Creating account…" : "Sign up"}
        </button>
      </form>
      <p className="mt-6 text-muted text-center">
        <Link className="cactus-link" to="/login">
          Already have an account?
        </Link>
      </p>
    </>
  );
}
