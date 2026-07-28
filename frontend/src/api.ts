/**
 * api.ts
 * ------
 * Every conversation with the Python backend goes through this one file.
 *
 * Keeping all the fetch calls together means that if the API ever moves or an
 * endpoint is renamed, there is exactly one place to fix it - and every caller
 * gets proper error handling for free.
 */

import type {
  AppConfig,
  BatchResult,
  DashboardData,
  DashboardFilters,
  OrderRequest,
  PredictionResult,
} from "./types";

/**
 * Ask the server for something, and turn any failure into a clear message.
 *
 * Without this wrapper, a failed request returns an HTML error page which then
 * explodes confusingly inside JSON.parse. Here we check first and throw
 * something a human can actually read.
 */
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(
      "Could not reach the server. Is the backend running on port 7860?"
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* the error body was not JSON - keep the status line */
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function fetchConfig(): Promise<AppConfig> {
  return request<AppConfig>("/api/config");
}

export function fetchDashboard(filters: DashboardFilters): Promise<DashboardData> {
  const params = new URLSearchParams();
  if (filters.monthFrom) params.set("monthFrom", filters.monthFrom);
  if (filters.monthTo) params.set("monthTo", filters.monthTo);
  // Repeating the same key is how FastAPI expects to receive a list.
  filters.states.forEach((s) => params.append("states", s));
  filters.categories.forEach((c) => params.append("categories", c));
  filters.payments.forEach((p) => params.append("payments", p));

  return request<DashboardData>(`/api/dashboard?${params.toString()}`);
}

export function predictOrder(order: OrderRequest): Promise<PredictionResult> {
  return request<PredictionResult>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(order),
  });
}

/** Upload a CSV of orders and score every row in one request. */
export function predictBatch(file: File): Promise<BatchResult> {
  const form = new FormData();
  form.append("file", file);
  // Deliberately no Content-Type header: the browser must set it itself so it
  // can append the multipart boundary token. Setting it by hand breaks upload.
  return request<BatchResult>("/api/predict/batch", {
    method: "POST",
    body: form,
  });
}

export const BATCH_TEMPLATE_URL = "/api/predict/batch/template";
