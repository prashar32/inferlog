import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getErrors, getRecentLogs, getSummary, getTimeseries } from "../api";
import type {
  ErrorsResponse,
  LogRow,
  MetricsSummary,
  TimeseriesPoint,
} from "../types";

const WINDOWS = [
  { label: "15m", minutes: 15, bucket: 60 },
  { label: "1h", minutes: 60, bucket: 60 },
  { label: "6h", minutes: 360, bucket: 300 },
  { label: "24h", minutes: 1440, bucket: 1800 },
];

const REFRESH_MS = 5000;

const fmtTime = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const fmtInt = (n: number) => n.toLocaleString();

export default function DashboardView() {
  const [windowIdx, setWindowIdx] = useState(1); // default 1h
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [points, setPoints] = useState<TimeseriesPoint[]>([]);
  const [errors, setErrors] = useState<ErrorsResponse | null>(null);
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [failed, setFailed] = useState(false);

  const win = WINDOWS[windowIdx];

  const load = useCallback(async () => {
    try {
      const [s, ts, er, lg] = await Promise.all([
        getSummary(win.minutes),
        getTimeseries(win.minutes, win.bucket),
        getErrors(win.minutes),
        getRecentLogs(12),
      ]);
      setSummary(s);
      setPoints(ts.points);
      setErrors(er);
      setLogs(lg);
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, [win.minutes, win.bucket]);

  useEffect(() => {
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => clearInterval(timer);
  }, [load]);

  const chartData = points.map((p) => ({ ...p, t: fmtTime(p.bucket) }));
  const successRate =
    summary && summary.total_requests > 0
      ? (summary.success / summary.total_requests) * 100
      : 100;

  return (
    <div className="dashboard">
      <div className="dashboard-head">
        <div>
          <h2>Inference observability</h2>
          <span className="chat-subtitle">
            auto-refreshing every {REFRESH_MS / 1000}s
          </span>
        </div>
        <div className="window-picker">
          {WINDOWS.map((w, i) => (
            <button
              key={w.label}
              className={i === windowIdx ? "chip active" : "chip"}
              onClick={() => setWindowIdx(i)}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {failed && (
        <div className="banner error">
          Could not reach the ingestion API. Is it running?
        </div>
      )}

      <div className="stat-grid">
        <Stat label="Requests" value={summary ? fmtInt(summary.total_requests) : "—"} />
        <Stat label="Success rate" value={`${successRate.toFixed(1)}%`} />
        <Stat
          label="Latency p95"
          value={summary ? `${fmtInt(summary.p95_latency_ms)} ms` : "—"}
          hint={summary ? `p50 ${summary.p50_latency_ms} · p99 ${summary.p99_latency_ms}` : ""}
        />
        <Stat
          label="Avg time-to-first-token"
          value={summary ? `${fmtInt(summary.avg_ttft_ms)} ms` : "—"}
        />
        <Stat label="Tokens" value={summary ? fmtInt(summary.total_tokens) : "—"} />
        <Stat
          label="Est. cost"
          value={summary ? `$${summary.total_cost_usd.toFixed(4)}` : "—"}
        />
      </div>

      <div className="chart-grid">
        <Panel title="Throughput — requests per bucket">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e6e8ee" />
              <XAxis dataKey="t" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="requests"
                stroke="#4f46e5"
                fill="#c7d2fe"
                name="requests"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Latency — avg vs p95 (ms)">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e6e8ee" />
              <XAxis dataKey="t" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="avg_latency_ms"
                stroke="#0ea5e9"
                name="avg"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="p95_latency_ms"
                stroke="#f59e0b"
                name="p95"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Errors & cancellations per bucket">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e6e8ee" />
              <XAxis dataKey="t" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Bar dataKey="errors" fill="#ef4444" name="errors" stackId="a" />
              <Bar dataKey="cancelled" fill="#a3a3a3" name="cancelled" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Errors by type">
          {errors && errors.by_type.length > 0 ? (
            <ul className="kv-list">
              {errors.by_type.map((e) => (
                <li key={e.error_type}>
                  <span>{e.error_type}</span>
                  <span className="kv-value">{e.count}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No errors in this window. 🎉</p>
          )}
        </Panel>
      </div>

      <Panel title="By model">
        <table className="data-table">
          <thead>
            <tr>
              <th>Provider</th>
              <th>Model</th>
              <th>Requests</th>
              <th>Errors</th>
              <th>Avg latency</th>
              <th>Tokens</th>
              <th>Est. cost</th>
            </tr>
          </thead>
          <tbody>
            {summary && summary.by_model.length > 0 ? (
              summary.by_model.map((r) => (
                <tr key={`${r.provider}/${r.model}`}>
                  <td>{r.provider}</td>
                  <td>{r.model}</td>
                  <td>{fmtInt(r.requests)}</td>
                  <td>{fmtInt(r.errors)}</td>
                  <td>{fmtInt(r.avg_latency_ms)} ms</td>
                  <td>{fmtInt(r.tokens)}</td>
                  <td>${r.cost_usd.toFixed(4)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={7} className="muted">
                  No inference logs yet — send a few chat messages.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>

      <Panel title="Recent inference logs (previews are PII-redacted)">
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Model</th>
              <th>Status</th>
              <th>Latency</th>
              <th>Tokens</th>
              <th>PII</th>
              <th>Input preview</th>
            </tr>
          </thead>
          <tbody>
            {logs.length > 0 ? (
              logs.map((l) => (
                <tr key={l.request_id}>
                  <td>{fmtTime(l.started_at)}</td>
                  <td>{l.model}</td>
                  <td>
                    <span className={`status-dot ${l.status}`} />
                    {l.status}
                  </td>
                  <td>{fmtInt(l.latency_ms)} ms</td>
                  <td>{l.total_tokens ?? "—"}</td>
                  <td>
                    {l.pii_redaction_count > 0 ? (
                      <span className="tag cancelled">
                        {l.pii_redaction_count} redacted
                      </span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="preview-cell">{l.input_preview ?? "—"}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={7} className="muted">
                  No inference logs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="panel">
      <div className="panel-title">{title}</div>
      {children}
    </div>
  );
}
