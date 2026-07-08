import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "@/api/client";

export function VerifyEmailPage() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("Missing verification token.");
      return;
    }
    api
      .post<{ success: boolean; error_message?: string }>(
        "/authentication/verify-email",
        { signup_token: token },
        false,
      )
      .then((res) => {
        if (res.success) {
          setStatus("ok");
        } else {
          setStatus("error");
          setMessage(res.error_message ?? "Verification failed");
        }
      })
      .catch((e) => {
        setStatus("error");
        setMessage(e instanceof Error ? e.message : "Verification failed");
      });
  }, [token]);

  return (
    <>
      <h1 className="title mb-4 text-2xl">Email verification</h1>
      {status === "loading" && <p className="text-muted">Verifying…</p>}
      {status === "ok" && (
        <>
          <p className="text-accent mb-6">Your email has been verified.</p>
          <Link className="btn-primary" to="/login">
            Log in
          </Link>
        </>
      )}
      {status === "error" && <p className="text-red-500">{message}</p>}
    </>
  );
}
