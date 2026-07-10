import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/api/client";

export function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await api.post("/authentication/forgot-password", { email }, false);
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    }
  };

  if (done) {
    return (
      <>
        <h1 className="title mb-4 text-2xl">Check your email</h1>
        <p className="text-muted">If an account exists, you will receive a reset link.</p>
        <Link className="cactus-link mt-6 inline-block" to="/login">
          Back to login
        </Link>
      </>
    );
  }

  return (
    <>
      <h1 className="title mb-8 text-2xl">Forgot password</h1>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="block text-xs text-muted font-semibold  mb-1">Email</label>
          <input className="field-input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary w-full">
          Send reset link
        </button>
      </form>
    </>
  );
}
