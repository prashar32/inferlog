export interface ModelOption {
  provider: string;
  model: string;
  label: string;
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  provider: string;
  model: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  status: string;
  request_id: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string | null;
  provider: string;
  model: string;
  system_prompt: string | null;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export interface ByModelRow {
  provider: string;
  model: string;
  requests: number;
  errors: number;
  avg_latency_ms: number;
  tokens: number;
  cost_usd: number;
}

export interface MetricsSummary {
  window_minutes: number;
  total_requests: number;
  success: number;
  errors: number;
  cancelled: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  avg_ttft_ms: number;
  total_tokens: number;
  total_cost_usd: number;
  by_model: ByModelRow[];
}

export interface TimeseriesPoint {
  bucket: string;
  requests: number;
  errors: number;
  cancelled: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  tokens: number;
}

export interface TimeseriesResponse {
  window_minutes: number;
  bucket_seconds: number;
  points: TimeseriesPoint[];
}

export interface LogRow {
  request_id: string;
  provider: string;
  model: string;
  status: string;
  streamed: boolean;
  latency_ms: number;
  ttft_ms: number | null;
  total_tokens: number | null;
  estimated_cost_usd: number | null;
  error_type: string | null;
  input_preview: string | null;
  pii_redaction_count: number;
  started_at: string;
}

export interface ErrorsResponse {
  window_minutes: number;
  by_type: { error_type: string; count: number }[];
  recent: {
    request_id: string;
    provider: string;
    model: string;
    error_type: string | null;
    error_message: string | null;
    started_at: string;
  }[];
}
