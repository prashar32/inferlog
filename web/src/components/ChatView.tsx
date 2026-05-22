import { useEffect, useRef, useState } from "react";
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  listModels,
  streamMessage,
} from "../api";
import type {
  ConversationDetail,
  ConversationSummary,
  Message,
  ModelOption,
} from "../types";
import ConversationList from "./ConversationList";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface Banner {
  kind: "error" | "info";
  text: string;
}

export default function ChatView() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [newModel, setNewModel] = useState("");

  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);

  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const [pendingUser, setPendingUser] = useState<string | null>(null);
  const [banner, setBanner] = useState<Banner | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const threadEndRef = useRef<HTMLDivElement | null>(null);

  // Initial load: models + conversations.
  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        if (m.length > 0) setNewModel(m[0].model);
      })
      .catch(() => setBanner({ kind: "error", text: "Could not load models." }));
    refreshConversations();
  }, []);

  // Keep the latest message in view.
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [detail, streamedText, pendingUser]);

  function refreshConversations() {
    listConversations().then(setConversations).catch(() => undefined);
  }

  function openConversation(id: string) {
    if (streaming) return; // don't switch mid-stream
    setBanner(null);
    setActiveId(id);
    getConversation(id)
      .then(setDetail)
      .catch(() => setBanner({ kind: "error", text: "Could not load conversation." }));
  }

  async function startNew() {
    if (streaming || !newModel) return;
    setBanner(null);
    try {
      const created = await createConversation(newModel);
      setActiveId(created.id);
      setDetail(created);
      refreshConversations();
    } catch {
      setBanner({ kind: "error", text: "Could not start a new conversation." });
    }
  }

  async function removeConversation(id: string) {
    try {
      await deleteConversation(id);
      if (id === activeId) {
        setActiveId(null);
        setDetail(null);
      }
      refreshConversations();
    } catch {
      setBanner({ kind: "error", text: "Could not delete conversation." });
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function send() {
    const text = input.trim();
    if (!text || streaming) return;
    setBanner(null);

    // Auto-create a conversation if the user just starts typing.
    let convId = activeId;
    if (!convId) {
      if (!newModel) return;
      try {
        const created = await createConversation(newModel);
        convId = created.id;
        setActiveId(created.id);
        setDetail(created);
      } catch {
        setBanner({ kind: "error", text: "Could not start a conversation." });
        return;
      }
    }

    setInput("");
    setPendingUser(text);
    setStreamedText("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    let aborted = false;

    try {
      await streamMessage(
        convId,
        text,
        {
          onToken: (t) => setStreamedText((s) => s + t),
          onDone: () => undefined,
          onError: (msg) => setBanner({ kind: "error", text: msg }),
        },
        controller.signal,
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        aborted = true;
        setBanner({ kind: "info", text: "Generation cancelled." });
      } else {
        setBanner({ kind: "error", text: "The response stream failed." });
      }
    } finally {
      abortRef.current = null;
      // On cancel the gateway persists the partial answer in the background;
      // give it a moment before re-syncing the thread from the server.
      if (aborted) await sleep(400);
      try {
        setDetail(await getConversation(convId));
      } catch {
        /* keep whatever we have */
      }
      setPendingUser(null);
      setStreamedText("");
      setStreaming(false);
      refreshConversations();
    }
  }

  const activeModel = detail?.model ?? newModel;

  return (
    <div className="chat-layout">
      <aside className="sidebar">
        <div className="sidebar-head">
          <label className="field-label">Model for new chats</label>
          <select
            value={newModel}
            onChange={(e) => setNewModel(e.target.value)}
            disabled={streaming}
          >
            {models.map((m) => (
              <option key={m.model} value={m.model}>
                {m.label}
              </option>
            ))}
          </select>
          <button className="primary-button" onClick={startNew} disabled={streaming}>
            + New chat
          </button>
        </div>
        <ConversationList
          conversations={conversations}
          activeId={activeId}
          onSelect={openConversation}
          onDelete={removeConversation}
        />
      </aside>

      <section className="chat-main">
        <div className="chat-head">
          <div>
            <h2>{detail?.title ?? "New conversation"}</h2>
            <span className="chat-subtitle">
              {activeId ? `model: ${activeModel}` : "pick a model and say hello"}
            </span>
          </div>
        </div>

        {banner && <div className={`banner ${banner.kind}`}>{banner.text}</div>}

        <div className="thread">
          {!detail && !pendingUser && (
            <div className="thread-empty">
              <p>Send a message to start.</p>
              <p className="muted">
                Every reply is logged through the inferlog SDK — open the
                Dashboard tab to watch it land.
              </p>
            </div>
          )}

          {detail?.messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}

          {pendingUser && (
            <div className="bubble user">
              <div className="bubble-role">you</div>
              <div className="bubble-body">{pendingUser}</div>
            </div>
          )}

          {streaming && (
            <div className="bubble assistant">
              <div className="bubble-role">assistant</div>
              <div className="bubble-body">
                {streamedText || <span className="cursor">▍</span>}
              </div>
            </div>
          )}

          <div ref={threadEndRef} />
        </div>

        <div className="composer">
          <textarea
            value={input}
            placeholder="Type a message…"
            rows={2}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          {streaming ? (
            <button className="stop-button" onClick={stop}>
              ■ Stop
            </button>
          ) : (
            <button
              className="primary-button"
              onClick={send}
              disabled={!input.trim()}
            >
              Send
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  return (
    <div className={`bubble ${message.role}`}>
      <div className="bubble-role">
        {message.role}
        {message.status === "cancelled" && (
          <span className="tag cancelled">cancelled</span>
        )}
        {message.status === "error" && <span className="tag error">error</span>}
      </div>
      <div className="bubble-body">
        {message.content || <span className="muted">(no content)</span>}
      </div>
      {(message.prompt_tokens != null || message.completion_tokens != null) && (
        <div className="bubble-meta">
          {message.prompt_tokens ?? 0} in · {message.completion_tokens ?? 0} out
        </div>
      )}
    </div>
  );
}
