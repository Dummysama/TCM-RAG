export const TOKEN_KEY = "tcm_rag_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

export async function apiFetch(path: string, options: RequestInit = {}) {
  const token = getToken();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(path, {
    ...options,
    headers,
  });

  const text = await res.text();
  let data: any = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!res.ok) {
    const msg =
      (data && typeof data === "object" && data.detail) ||
      `请求失败: ${res.status}`;
    throw new Error(msg);
  }

  return data;
}

export async function register(username: string, password: string) {
  return apiFetch("/api/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function login(username: string, password: string) {
  return apiFetch("/api/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function getMe() {
  return apiFetch("/api/me");
}

export async function getConversations() {
  return apiFetch("/api/conversations");
}

export async function createConversationApi(title = "新会话") {
  return apiFetch("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function getConversationMessages(conversationId: number) {
  return apiFetch(`/api/conversations/${conversationId}/messages`);
}

export async function askInConversation(conversationId: number, question: string) {
  return apiFetch(`/api/conversations/${conversationId}/ask`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}