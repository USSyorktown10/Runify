import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "@/api/client";

export function ResetPasswordPage() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const [password, setPassword] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await api.post<{ success: boolean; error_message?: string }>(
        "/authentication/reset-password",
        { reset_token: token, new_password: password },
        false,
      );
      if (!res.success) {
        setError(res.error_message ?? "Reset failed");
        return;
      }
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reset failed");
    }
  };

  if (!token) {
    return <p className="text-red-500">Invalid reset link.</p>;
  }

  if (done) {
    return (
      <>
        <h1 className="title mb-4 text-2xl">Password updated</h1>
        <Link className="btn-primary" to="/login">
          Log in
        </Link>
      </>
    );
  }

  return (
    <>
      <h1 className="title mb-8 text-2xl">Reset password</h1>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="block text-xs text-muted font-semibold uppercase mb-1">New password</label>
          <input
            className="field-input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        {error && <p className="text-red-500">{error}</p>}
        <button type="submit" className="btn-primary w-full">
          Update password
        </button>
      </form>
    </>
  );
}
