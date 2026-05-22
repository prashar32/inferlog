import type { ConversationSummary } from "../types";

interface Props {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function ConversationList({
  conversations,
  activeId,
  onSelect,
  onDelete,
}: Props) {
  if (conversations.length === 0) {
    return <p className="sidebar-empty">No conversations yet.</p>;
  }

  return (
    <ul className="conversation-list">
      {conversations.map((c) => (
        <li
          key={c.id}
          className={c.id === activeId ? "conversation active" : "conversation"}
          onClick={() => onSelect(c.id)}
        >
          <div className="conversation-main">
            <div className="conversation-title">
              {c.title ?? "New conversation"}
            </div>
            <div className="conversation-sub">
              {c.last_message
                ? c.last_message.slice(0, 48)
                : "no messages yet"}
            </div>
          </div>
          <div className="conversation-meta">
            <span className="badge">{c.model}</span>
            <button
              className="icon-button"
              title="Delete conversation"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(c.id);
              }}
            >
              ×
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
