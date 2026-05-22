import { useState } from "react";
import ChatView from "./components/ChatView";
import DashboardView from "./components/DashboardView";

type Tab = "chat" | "dashboard";

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◆</span>
          <span>InferLog</span>
        </div>
        <nav className="tabs">
          <button
            className={tab === "chat" ? "tab active" : "tab"}
            onClick={() => setTab("chat")}
          >
            Chat
          </button>
          <button
            className={tab === "dashboard" ? "tab active" : "tab"}
            onClick={() => setTab("dashboard")}
          >
            Dashboard
          </button>
        </nav>
        <div className="topbar-note">inference logging &amp; ingestion</div>
      </header>
      <main className="content">
        {tab === "chat" ? <ChatView /> : <DashboardView />}
      </main>
    </div>
  );
}
