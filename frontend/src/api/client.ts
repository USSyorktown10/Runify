const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

let tokenGetter: (() => string | null) | null = null;
let onUnauthorized: (() => void) | null = null;

export function setTokenGetter(getter: () => string | null) {
  tokenGetter = getter;
}

export function setOnUnauthorized(handler: () => void) {
  onUnauthorized = handler;
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function formatErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "object" && item && "msg" in item) {
          return String((item as { msg: string }).msg);
        }
        return typeof item === "string" ? item : null;
      })
      .filter(Boolean);
    if (messages.length > 0) return messages.join("; ");
  }
  if (typeof detail === "object" && detail && "msg" in detail) {
    return String((detail as { msg: string }).msg);
  }
  return fallback;
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  auth = true,
): Promise<T> {
  const headers = new Headers(options.headers);
  const hasBody = options.body !== undefined && options.body !== null;
  if (hasBody && !headers.has("Content-Type") && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (auth && tokenGetter) {
    const token = tokenGetter();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (res.status === 401 && onUnauthorized) {
    onUnauthorized();
  }

  if (!res.ok) {
    let message = res.statusText;
    try {
      const errBody = await res.json();
      message = formatErrorMessage(
        errBody.detail ?? errBody.error_message,
        message,
      );
    } catch {
      /* ignore */
    }
    throw new ApiError(message, res.status);
  }

  if (res.status === 204) return undefined as T;

  const contentType = res.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return res.json() as Promise<T>;
  }
  return res as unknown as T;
}

export const api = {
  get: <T>(path: string, auth = true) => request<T>(path, { method: "GET" }, auth),
  post: <T>(path: string, body?: unknown, auth = true) =>
    request<T>(
      path,
      {
        method: "POST",
        body:
          body === undefined
            ? undefined
            : body instanceof FormData
              ? body
              : JSON.stringify(body),
      },
      auth,
    ),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  download: async (path: string) => {
    const headers = new Headers();
    if (tokenGetter) {
      const token = tokenGetter();
      if (token) headers.set("Authorization", `Bearer ${token}`);
    }
    const res = await fetch(`${BASE}${path}`, { headers });
    if (!res.ok) throw new ApiError(res.statusText, res.status);
    return res.blob();
  },
};
