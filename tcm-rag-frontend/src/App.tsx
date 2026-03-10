import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Plus, Search, Trash2, Send, Edit3, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import "./index.css";
import {
  clearToken,
  getMe,
  login,
  register,
  setToken,
  getConversations,
  createConversationApi,
  getConversationMessages,
  askInConversation,
} from "./auth";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  intent?: string;
  entity?: string | null;
  references?: string[];
  reference_items?: { id: string; type?: string; name: string }[];
};

type Conversation = {
  id: number;
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
  reference_items?: { id: string; type?: string; name: string }[];
};

type HerbDetail = {
  id: string;
  name?: string;
  pinyin?: string;
  latin_name?: string;
  english_name?: string;
  nature?: string;
  flavor?: string;
  meridian?: string;
  effects?: string;
  indications?: string;
  alias?: string;
  raw_text?: string;
};


const uid = () => Math.random().toString(36).slice(2, 10);
const now = () => new Date().toISOString();

function buildConversation(title = "新主题会话", theme = "默认主题"): Conversation {
  const t = now();
  return {
    id: Date.now(),
    title,
    theme,
    createdAt: t,
    updatedAt: t,
    messages: [
      {
        id: uid(),
        role: "assistant",
        content: "你好，这里是中医药 RAG 问答平台。",
        createdAt: t,
      },
    ],
  };
}

function buildInitialConversations(): Conversation[] {
  return [buildConversation()];
}

function toConversation(raw: any): Conversation {
  return {
    id: raw.id,
    title: raw.title || "新会话",
    theme: "默认主题",
    createdAt: raw.created_at || now(),
    updatedAt: raw.updated_at || now(),
    messages: [],
  };
}

function toMessage(raw: any): Message {
  let references: string[] = [];
  try {
    references = raw.references_json ? JSON.parse(raw.references_json) : [];
  } catch {
    references = [];
  }

  return {
    id: String(raw.id),
    role: raw.role,
    content: raw.content,
    createdAt: raw.created_at || now(),
    intent: raw.intent,
    entity: raw.entity,
    references,
    reference_items: [],
  };
}



function summarizeTitle(input: string) {
  const clean = input.replace(/\s+/g, " ").trim();
  return clean.length <= 16 ? clean : clean.slice(0, 16) + "…";
}

const apiClient = {
  async ask(question: string, conversation: Conversation): Promise<AskResponse> {
    const res = await fetch("/api/ask", {
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

  async getHerbDetail(id: string): Promise<HerbDetail> {
    const herbId = id.startsWith("herb::") ? id.replace("herb::", "") : id;
    const res = await fetch(`/api/herb/${herbId}`);
    if (!res.ok) {
      throw new Error(`中药详情请求失败: ${res.status}`);
    }
    return await res.json();
  },
};

function MessageBubble({
  message,
  onOpenHerbDetail,
}: {
  message: Message;
  onOpenHerbDetail: (id: string) => void;
}) {
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

            {message.reference_items?.length ? (
                  <div className="message-badges">
                    {message.reference_items
                      .filter((item) => item.type === "herb")
                      .map((item) => (
                        <button
                          key={item.id}
                          className="badge badge-outline badge-clickable"
                          onClick={() => onOpenHerbDetail(item.id)}
                          type="button"
                        >
                          {item.name}
                        </button>
                      ))}
                  </div>
                ) : message.references?.length ? (
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

  const initialConversations = buildInitialConversations();

  const [conversations, setConversations] = useState<Conversation[]>(initialConversations);
  const [activeId, setActiveId] = useState<number>(initialConversations[0].id);
  const [query, setQuery] = useState("");

  const [search, setSearch] = useState("");
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [herbDetail, setHerbDetail] = useState<HerbDetail | null>(null);

  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authLoading, setAuthLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<{ id: number; username: string } | null>(null);

  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");


//加载会话列表
    const loadUserConversations = async () => {
  try {
    const rows = await getConversations();

    if (!Array.isArray(rows) || rows.length === 0) {
      const created = await createConversationApi("新会话");
      const conv = toConversation(created);
      setConversations([conv]);
      setActiveId(conv.id);
      return;
    }

    const mapped = rows.map(toConversation);
    const firstId = mapped[0].id;

    const msgRows = await getConversationMessages(firstId);
    const msgMapped = Array.isArray(msgRows) ? msgRows.map(toMessage) : [];

    mapped[0] = {
      ...mapped[0],
      messages: msgMapped,
    };

    setConversations(mapped);
    setActiveId(firstId);
  } catch (error) {
    console.error("加载会话失败:", error);
    const fresh = buildInitialConversations();
    setConversations(fresh);
    setActiveId(fresh[0].id);
  }
};

  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
  if (window.innerWidth <= 768) {
    document.body.style.overflow = sidebarOpen ? "hidden" : "";
  }
  return () => {
    document.body.style.overflow = "";
  };
}, [sidebarOpen]);

        useEffect(() => {
          const bootstrapAuth = async () => {
            try {
              const me = await getMe();
              setCurrentUser(me);
              setIsAuthenticated(true);
              await loadUserConversations();
            } catch {
              setIsAuthenticated(false);
              setCurrentUser(null);
            } finally {
              setAuthLoading(false);
            }
          };

          bootstrapAuth();
        }, []);

    const handleAuthSubmit = async () => {
      const username = authUsername.trim();
      const password = authPassword.trim();

      if (!username || !password) {
        setAuthError("用户名和密码不能为空");
        return;
      }

      try {
        setAuthError("");

        if (authMode === "register") {
          await register(username, password);
        }

        const loginResp = await login(username, password);
        setToken(loginResp.access_token);

        const me = await getMe();
        setCurrentUser(me);
        setIsAuthenticated(true);

        setQuery("");
        setSearch("");
        await loadUserConversations();

        setAuthUsername("");
        setAuthPassword("");
      } catch (error: any) {
        setAuthError(error.message || "操作失败");
      }
    };

    const handleLogout = () => {
      clearToken();
      setIsAuthenticated(false);
      setCurrentUser(null);

      // 清空当前前端本地聊天状态，避免不同账户看到旧记录
      const fresh = buildInitialConversations();
      setConversations(fresh);
      setActiveId(fresh[0].id);
      setQuery("");
      setSearch("");
    };

  const openHerbDetail = async (id: string) => {
  try {
    setDetailOpen(true);
    setDetailLoading(true);
    const detail = await apiClient.getHerbDetail(id);
    setHerbDetail(detail);
  } catch (error) {
    console.error("openHerbDetail crashed:", error);
    setHerbDetail({
      id,
      name: "详情加载失败",
      raw_text: "",
    });
  } finally {
    setDetailLoading(false);
  }
};

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
    }
  }, [activeConversation]);

//   useEffect(() => {
//     saveState(conversations, activeId);
//   }, [conversations, activeId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversation?.messages.length, sending]);

  const updateConversation = (id: number, updater: (old: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  };

const createConversation = async () => {
  try {
    const row = await createConversationApi("新会话");
    const conv = toConversation(row);

    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setSidebarOpen(false);
  } catch (error) {
    console.error("创建会话失败:", error);
  }
};

  const deleteConversation = (id: number) => {
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

  const renameConversation = (id: number) => {
    const nextTitle = window.prompt("请输入新的会话标题");
    if (!nextTitle?.trim()) return;

    updateConversation(id, (old) => ({
      ...old,
      title: nextTitle.trim(),
      updatedAt: now(),
    }));
  };

const openConversation = async (id: number) => {
  try {
    setActiveId(id);
    setSidebarOpen(false);

    const rows = await getConversationMessages(id);
    const mapped = Array.isArray(rows) ? rows.map(toMessage) : [];

    setConversations((prev) =>
      prev.map((c) =>
        c.id === id
          ? {
              ...c,
              messages: mapped,
            }
          : c
      )
    );
  } catch (error) {
    console.error("加载会话消息失败:", error);
  }
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

      const resp = await askInConversation(activeConversation.id, userText);

const rows = await getConversationMessages(activeConversation.id);
const mapped = Array.isArray(rows) ? rows.map(toMessage) : [];

setConversations((prev) =>
  prev.map((c) =>
    c.id === activeConversation.id
      ? {
          ...c,
          title:
            typeof resp.title === "string" && resp.title.trim().length > 0
              ? resp.title.trim()
              : c.title,
          updatedAt: now(),
          messages: mapped,
        }
      : c
  )
);
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

if (authLoading) {
  return (
    <div className="auth-page">
      <div className="auth-card">
        <h2>正在验证登录状态...</h2>
      </div>
    </div>
  );
}

if (!isAuthenticated) {
  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>TCM-RAG</h1>
        <p className="auth-subtitle">登录后即可保存个人咨询记录</p>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${authMode === "login" ? "active" : ""}`}
            onClick={() => setAuthMode("login")}
          >
            登录
          </button>
          <button
            className={`auth-tab ${authMode === "register" ? "active" : ""}`}
            onClick={() => setAuthMode("register")}
          >
            注册
          </button>
        </div>

        <input
          className="auth-input"
          value={authUsername}
          onChange={(e) => setAuthUsername(e.target.value)}
          placeholder="请输入用户名"
        />

        <input
          className="auth-input"
          type="password"
          value={authPassword}
          onChange={(e) => setAuthPassword(e.target.value)}
          placeholder="请输入密码"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleAuthSubmit();
            }
          }}
        />

        {authError ? <div className="auth-error">{authError}</div> : null}

        <button className="auth-submit" onClick={handleAuthSubmit}>
          {authMode === "login" ? "登录" : "注册并登录"}
        </button>
      </div>
    </div>
  );
}

  return (
    <div className="app-shell">
      <header className="topbar">
          <button className="icon-toggle" onClick={() => setSidebarOpen((v) => !v)}>
            {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
          </button>

          <div className="topbar-title">{activeConversation.title}</div>

          <div className="topbar-user">
              <div className="avatar">
                {currentUser?.username?.[0]?.toUpperCase()}
              </div>

              <span className="topbar-username">
                {currentUser?.username}
              </span>

              <button className="logout-btn" onClick={handleLogout}>
                退出
              </button>
            </div>
        </header>



      <div className="layout">
        <>
          <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
            <div className="sidebar-inner">
              <button
                className="btn btn-primary new-btn"
                onClick={() => {
                  createConversation();
                  setSidebarOpen(false);
                }}
              >
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
                    <button
                      className="conversation-main"
                      onClick={() => openConversation(c.id)}
                    >
                      <div className="conversation-title">{c.title}</div>
                      <div className="conversation-theme">{c.theme}</div>
                    </button>

                    <div className="conversation-actions">
                      <button
                        className="btn btn-light btn-small"
                        onClick={() => renameConversation(c.id)}
                      >
                        <Edit3 size={13} />
                      </button>
                      <button
                        className="btn btn-light btn-small"
                        onClick={() => deleteConversation(c.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          {sidebarOpen ? (
            <div
              className="sidebar-backdrop"
              onClick={() => setSidebarOpen(false)}
            />
          ) : null}
        </>

        <main className="main">
          <section className="messages-panel">
            <div className="messages">
              {activeConversation.messages.map((m) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    onOpenHerbDetail={openHerbDetail}
                  />
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
    {detailOpen ? (
          <div className="detail-overlay" onClick={() => setDetailOpen(false)}>
            <div className="detail-modal" onClick={(e) => e.stopPropagation()}>
              <button className="detail-close" onClick={() => setDetailOpen(false)}>
                ×
              </button>

              {detailLoading ? (
                <div className="detail-loading">正在加载中药详情...</div>
              ) : herbDetail ? (
                <div className="detail-content">
                  <h3>{herbDetail.name || "中药详情"}</h3>
                  {herbDetail.pinyin ? <p><strong>拼音：</strong>{herbDetail.pinyin}</p> : null}
                  {herbDetail.latin_name ? <p><strong>拉丁名：</strong>{herbDetail.latin_name}</p> : null}
                  {herbDetail.english_name ? <p><strong>英文名：</strong>{herbDetail.english_name}</p> : null}
                  {herbDetail.alias ? <p><strong>别名：</strong>{herbDetail.alias}</p> : null}
                  {herbDetail.nature ? <p><strong>性：</strong>{herbDetail.nature}</p> : null}
                  {herbDetail.flavor ? <p><strong>味：</strong>{herbDetail.flavor}</p> : null}
                  {herbDetail.meridian ? <p><strong>归经：</strong>{herbDetail.meridian}</p> : null}
                  {herbDetail.effects ? <p><strong>功效：</strong>{herbDetail.effects}</p> : null}
                  {herbDetail.indications ? <p><strong>主治：</strong>{herbDetail.indications}</p> : null}
                </div>
              ) : (
                <div className="detail-loading">暂无详情</div>
              )}
            </div>
          </div>
        ) : null}
    </div>
  );
}
