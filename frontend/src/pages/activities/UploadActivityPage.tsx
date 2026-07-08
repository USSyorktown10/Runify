import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";

export function UploadActivityPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const startUpload = async () => {
    if (!file) return;
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await api.post<{ success: boolean; upload_id?: string; error_message?: string }>("/upload", form);
      if (!res.success || !res.upload_id) {
        setError(res.error_message ?? "Upload failed");
        return;
      }
      setUploadId(res.upload_id);
      setStatus("processing");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  useEffect(() => {
    if (!uploadId) return;
    const interval = setInterval(async () => {
      try {
        const res = await api.get<{ status: string; activity_id?: string; error_message?: string }>(
          `/upload/${uploadId}`,
        );
        setStatus(res.status);
        if (res.activity_id) {
          clearInterval(interval);
          navigate(`/activities/${res.activity_id}`);
        }
        if (res.status === "failed") {
          clearInterval(interval);
          setError(res.error_message ?? "Processing failed");
        }
      } catch {
        /* retry */
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [uploadId, navigate]);

  return (
    <section>
      <h1 className="title mb-8">Upload activity</h1>
      <div className="max-w-md space-y-4">
        <input
          type="file"
          accept=".gpx,.fit,.tcx"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {status && <p className="text-muted">Status: {status}</p>}
        {error && <p className="text-red-500">{error}</p>}
        <button type="button" className="btn-primary" onClick={startUpload} disabled={!file || !!uploadId}>
          Upload
        </button>
      </div>
    </section>
  );
}
