import type {
  ConversationDetail,
  ConversationSummary,
  ErrorsResponse,
  LogRow,
  MetricsSummary,
  ModelOption,
  TimeseriesResponse,
} from "./types";

// Both paths are reverse-proxied — by nginx in Docker, by Vite in dev.
const GATEWAY = "/api/gateway";
const INGESTION = "/api/ingestion";

async function getJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json() as Promise<T>;
}

// ---- gateway: chat -----------------------------------------------------

export const listModels = () => getJSON<ModelOption[]>(`${GATEWAY}/v1/models`);

export const listConversations = () =>
  getJSON<ConversationSummary[]>(`${GATEWAY}/v1/conversations`);

export const getConversation = (id: string) =>
  getJSON<ConversationDetail>(`${GATEWAY}/v1/conversations/${id}`);

export async function createConversation(model: string): Promise<ConversationDetail> {
  const resp = await fetch(`${GATEWAY}/v1/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!resp.ok) throw new Error(`could not create conversation (${resp.status})`);
  return resp.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const resp = await fetch(`${GATEWAY}/v1/conversations/${id}`, { method: "DELETE" });
  if (!resp.ok && resp.status !== 404) {
    throw new Error(`could not delete conversation (${resp.status})`);
  }
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

/**
 * POST a message and consume the SSE stream. Pass an AbortSignal to support
 * the Stop button — aborting it cancels the request, which the gateway logs
 * as a `cancelled` inference.
 */
export async function streamMessage(
  conversationId: string,
  content: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch(
    `${GATEWAY}/v1/conversations/${conversationId}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
      signal,
    },
  );
  if (!resp.ok || !resp.body) {
    handlers.onError(`request failed (${resp.status})`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let split: number;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      dispatchFrame(frame, handlers);
    }
  }
  handlers.onDone();
}

function dispatchFrame(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  if (!data) return;
  const payload = JSON.parse(data);
  if (event === "token") handlers.onToken(payload.text);
  else if (event === "error") handlers.onError(payload.message);
}

// ---- ingestion: dashboards --------------------------------------------

export const getSummary = (windowMin: number) =>
  getJSON<MetricsSummary>(`${INGESTION}/v1/metrics/summary?window=${windowMin}`);

export const getTimeseries = (windowMin: number, bucketSec: number) =>
  getJSON<TimeseriesResponse>(
    `${INGESTION}/v1/metrics/timeseries?window=${windowMin}&bucket=${bucketSec}`,
  );

export const getErrors = (windowMin: number) =>
  getJSON<ErrorsResponse>(`${INGESTION}/v1/metrics/errors?window=${windowMin}`);

export const getRecentLogs = (limit: number) =>
  getJSON<LogRow[]>(`${INGESTION}/v1/logs?limit=${limit}`);
