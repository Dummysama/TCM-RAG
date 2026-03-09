import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Plus, Search, Trash2, Send, Edit3, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import "./index.css";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  intent?: string;
  entity?: string | null;
  references?: string[];
};

type Conversation = {
  id: string;
  title: string;
  theme: string;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
};

type AskResponse = {
  answer: string;
  intent?: string;
  entity?: string | null;
  references?: string[];
  title?: string;
};

const STORAGE_KEY = "tcm_rag_frontend_state_stable_v1";

const uid = () => Math.random().toString(36).slice(2, 10);
const now = () => new Date().toISOString();

function buildConversation(title = "新主题会话", theme = "默认主题"): Conversation {
  const t = now();
  return {
    id: uid(),
    title,
    theme,
    createdAt: t,
    updatedAt: t,
    messages: [
      {
        id: uid(),
        role: "assistant",
        content: "你好，这里是中医药 RAG 问答界面。",
        createdAt: t,
      },
    ],
  };
}

function saveState(conversations: Conversation[], activeId: string | null) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ conversations, activeId }));
}

function loadState(): { conversations: Conversation[]; activeId: string | null } {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    const seed = buildConversation();
    return { conversations: [seed], activeId: seed.id };
  }

  try {
    const parsed = JSON.parse(raw);
    if (!parsed?.conversations?.length) {
      const seed = buildConversation();
      return { conversations: [seed], activeId: seed.id };
    }
    return parsed;
  } catch {
    const seed = buildConversation();
    return { conversations: [seed], activeId: seed.id };
  }
}

function summarizeTitle(input: string) {
  const clean = input.replace(/\s+/g, " ").trim();
  return clean.length <= 16 ? clean : clean.slice(0, 16) + "…";
}

const apiClient = {
  async ask(question: string, conversation: Conversation): Promise<AskResponse> {
    const res = await fetch("http://127.0.0.1:8000/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        conversation_id: conversation.id,
        theme: conversation.theme,
        history: conversation.messages,
      }),
    });

    if (!res.ok) {
      throw new Error(`后端接口请求失败: ${res.status}`);
    }

    return await res.json();
  },
};

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={`message-row ${isUser ? "message-row-user" : "message-row-assistant"}`}
    >
      <div className={`message-bubble ${isUser ? "message-bubble-user" : "message-bubble-assistant"}`}>
        <div className="message-content">{message.content}</div>

        {!isUser && (message.intent || message.entity || message.references?.length) ? (
          <div className="message-meta">
            <div className="message-badges">
              {message.intent ? <span className="badge">intent: {message.intent}</span> : null}
              {message.entity ? <span className="badge badge-outline">entity: {message.entity}</span> : null}
            </div>

            {message.references?.length ? (
              <div className="message-badges">
                {message.references.map((ref) => (
                  <span key={ref} className="badge badge-outline">
                    {ref}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </motion.div>
  );
}

export default function App() {
  const initial = useMemo(() => loadState(), []);
  const [conversations, setConversations] = useState<Conversation[]>(initial.conversations);
  const [activeId, setActiveId] = useState<string | null>(initial.activeId);
  const [query, setQuery] = useState("");
  const [themeName, setThemeName] = useState("默认主题");
  const [search, setSearch] = useState("");
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const endRef = useRef<HTMLDivElement | null>(null);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? conversations[0] ?? null,
    [conversations, activeId]
  );

  const filteredConversations = useMemo(() => {
    const kw = search.trim();
    if (!kw) return conversations;
    return conversations.filter((c) => c.title.includes(kw) || c.theme.includes(kw));
  }, [conversations, search]);

  useEffect(() => {
    if (activeConversation) {
      setThemeName(activeConversation.theme);
    }
  }, [activeConversation]);

  useEffect(() => {
    saveState(conversations, activeId);
  }, [conversations, activeId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversation?.messages.length, sending]);

  const updateConversation = (id: string, updater: (old: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  };

  const createConversation = () => {
    const title = `主题会话 ${conversations.length + 1}`;
    const conv = buildConversation(title, themeName.trim() || "默认主题");
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  };

  const deleteConversation = (id: string) => {
    const next = conversations.filter((c) => c.id !== id);
    if (!next.length) {
      const seed = buildConversation();
      setConversations([seed]);
      setActiveId(seed.id);
      return;
    }
    setConversations(next);
    if (activeId === id) {
      setActiveId(next[0].id);
    }
  };

  const renameConversation = (id: string) => {
    const nextTitle = window.prompt("请输入新的会话标题");
    if (!nextTitle?.trim()) return;

    updateConversation(id, (old) => ({
      ...old,
      title: nextTitle.trim(),
      updatedAt: now(),
    }));
  };

  const send = async () => {
    try {
      if (!query.trim() || !activeConversation || sending) return;

      const userText = query.trim();
      setQuery("");
      setSending(true);

      const userMsg: Message = {
        id: uid(),
        role: "user",
        content: userText,
        createdAt: now(),
      };

      updateConversation(activeConversation.id, (old) => ({
        ...old,
        title: old.messages.length <= 1 ? summarizeTitle(userText) : old.title,
        theme: old.theme,
        updatedAt: now(),
        messages: [...old.messages, userMsg],
      }));

      const latestConv: Conversation = {
        ...activeConversation,
        messages: [...activeConversation.messages, userMsg],
      };

      const resp = await apiClient.ask(userText, latestConv);

      const assistantMsg: Message = {
        id: uid(),
        role: "assistant",
        content: typeof resp.answer === "string" ? resp.answer : "后端未返回有效回答。",
        createdAt: now(),
        intent: resp.intent,
        entity: resp.entity,
        references: Array.isArray(resp.references) ? resp.references : [],
      };

      updateConversation(activeConversation.id, (old) => ({
        ...old,
        title:
          typeof resp.title === "string" && resp.title.trim().length > 0
            ? resp.title.trim()
            : old.title,
        theme: themeName.trim() || old.theme,
        updatedAt: now(),
        messages: [...old.messages, assistantMsg],
      }));
    } catch (error) {
      console.error("send() crashed:", error);

      if (activeConversation) {
        const assistantMsg: Message = {
          id: uid(),
          role: "assistant",
          content: "前端渲染或接口处理出错，请查看浏览器控制台。",
          createdAt: now(),
        };

        updateConversation(activeConversation.id, (old) => ({
          ...old,
          updatedAt: now(),
          messages: [...old.messages, assistantMsg],
        }));
      }
    } finally {
      setSending(false);
    }
  };

  if (!activeConversation) {
    return <div style={{ color: "white", padding: 20 }}>当前没有可用会话。</div>;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="icon-toggle" onClick={() => setSidebarOpen((v) => !v)}>
          {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
        </button>
        <div className="topbar-title">{activeConversation.title}</div>
      </header>

      <div className="layout">
        <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
          <div className="sidebar-inner">
            <button className="btn btn-primary new-btn" onClick={createConversation}>
              <Plus size={16} />
              新建会话
            </button>

            <div className="search-box">
              <Search size={15} />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="搜索会话"
              />
            </div>


            <div className="conversation-list">
              {filteredConversations.map((c) => (
                <div
                  key={c.id}
                  className={`conversation-item ${c.id === activeConversation.id ? "active" : ""}`}
                >
                  <button className="conversation-main" onClick={() => setActiveId(c.id)}>
                    <div className="conversation-title">{c.title}</div>
                    <div className="conversation-theme">{c.theme}</div>
                  </button>

                  <div className="conversation-actions">
                    <button className="btn btn-light btn-small" onClick={() => renameConversation(c.id)}>
                      <Edit3 size={13} />
                    </button>
                    <button className="btn btn-light btn-small" onClick={() => deleteConversation(c.id)}>
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="main">
          <section className="messages-panel">
            <div className="messages">
              {activeConversation.messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}

              {sending ? (
                <div className="message-row message-row-assistant">
                  <div className="message-bubble message-bubble-assistant">正在生成回答...</div>
                </div>
              ) : null}

              <div ref={endRef} />
            </div>
          </section>

          <section className="composer-wrap">
            <div className="composer-panel">


              <div className="composer">
                    <textarea
                      className="textarea"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          if (!sending && query.trim()) {
                            send();
                          }
                        }
                      }}
                      placeholder="输入问题，例如：地耳草的功效是什么？或 咽喉肿痛、发热、口渴明显，有什么中药可参考？"
                    />
                <button className="send-btn" onClick={send} disabled={sending || !query.trim()}>
                  <Send size={16} />
                </button>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}