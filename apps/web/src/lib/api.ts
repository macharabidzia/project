export interface RuntimeReadinessCheck {
  ready: boolean;
  target: string;
  detail: string;
  status: "ready" | "failed" | "pending" | "running" | "skipped";
}

export interface RuntimeReadinessStatus {
  ready: boolean;
  session_eligible: boolean;
  status: "ready" | "failed" | "booting" | "checking" | "downloading" | "warming";
  progress: number;
  summary: string;
  checks: Record<string, RuntimeReadinessCheck>;
}

export interface RuntimeTelemetryStatus {
  available: boolean;
  summary: string;
  session_id: string;
  updated_at: number;
  health: Record<string, boolean>;
  metrics: Record<string, number | string | boolean>;
  stats: Record<string, number>;
}

export interface SystemConfig {
  app_name: string;
  app_env: string;
  transport: {
    kind: string;
    livekit_url: string;
    room_name: string;
    runtime_identity: string;
    output_track_name: string;
    turn_enabled: boolean;
  };
  layers: Array<{
    name: string;
    purpose: string;
    backend: string;
    status: string;
  }>;
}

const trimSlash = (value: string) => value.replace(/\/+$/, "");

const deriveApiBaseFromBrowser = () => {
  if (typeof window === "undefined") {
    return "";
  }
  const origin = window.location.origin;
  const host = window.location.host;
  if (host.endsWith("-5173.proxy.runpod.net")) {
    return trimSlash(origin.replace("-5173.proxy.runpod.net", "-8000.proxy.runpod.net"));
  }
  if (window.location.port === "5173") {
    return trimSlash(origin.replace(":5173", ":8000"));
  }
  return "";
};

export const resolveApiBaseUrl = () => {
  const configured = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.trim();
  if (configured) {
    return trimSlash(configured);
  }
  const derived = deriveApiBaseFromBrowser();
  if (derived) {
    return derived;
  }
  return "http://127.0.0.1:8000";
};

export const apiBaseUrl = resolveApiBaseUrl();

export interface LiveKitTokenResponse {
  url: string;
  room_name: string;
  identity: string;
  token: string;
  turn_enabled: boolean;
}

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`);
  if (!response.ok) {
    throw new Error(await response.text() || `Request failed with ${response.status}`);
  }
  return await response.json() as T;
}

export const api = {
  getSystemConfig: () => requestJson<SystemConfig>("/v1/system/config"),
  getRuntimeReadiness: () => requestJson<RuntimeReadinessStatus>("/v1/system/readiness"),
  getRuntimeTelemetry: () => requestJson<RuntimeTelemetryStatus>("/v1/system/runtime"),
  getLiveKitToken: (identity: string) => requestJson<LiveKitTokenResponse>(`/v1/livekit/token?identity=${encodeURIComponent(identity)}`),
};
