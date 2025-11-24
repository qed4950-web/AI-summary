import { useMemo, useRef, useState } from "react";
import "./App.css";

type SearchHit = {
  path: string;
  similarity?: number;
  preview?: string;
};

type SearchResponse = {
  answer?: string;
  results?: SearchHit[];
  session_id?: string;
  llm_error?: string | null;
};

type Message = {
  role: "user" | "assistant" | "system";
  text: string;
  hits?: SearchHit[];
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8080";

export default function App() {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "안녕하세요! 대화를 하려면 그냥 입력하고, 문서/오디오 요약 검색은 `/search ...`로 시작해 주세요.",
    },
  ]);
  const listRef = useRef<HTMLDivElement | null>(null);

  const apiUrl = useMemo(() => `${API_BASE.replace(/\/+$/, "")}/api/search`, []);

  async function send() {
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    appendMessage({ role: "user", text: trimmed });
    setError(null);

    const isSearch = trimmed.startsWith("/search") || trimmed.startsWith("/doc");
    if (!isSearch) {
      appendMessage({
        role: "assistant",
        text: "검색/요약이 필요하면 `/search`나 `/doc`으로 시작해주세요. 대화는 여기에 그대로 입력하시면 됩니다.",
      });
      return;
    }

    setLoading(true);
    try {
      const payload = {
        query: trimmed,
        top_k: 5,
        session_id: sessionId || undefined,
      };
      const resp = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const json = (await resp.json()) as SearchResponse;
      if (json.session_id) setSessionId(json.session_id);

      const hits = json.results || [];
      appendMessage({
        role: "assistant",
        text: json.answer || "결과가 없습니다.",
        hits,
      });
      if (json.llm_error) {
        appendMessage({
          role: "system",
          text: `LLM 오류: ${json.llm_error}`,
        });
      }
    } catch (exc: any) {
      setError(exc?.message || "요청 중 오류가 발생했습니다.");
      appendMessage({ role: "assistant", text: "요청 처리에 실패했습니다. 잠시 후 다시 시도해주세요." });
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }

  function appendMessage(msg: Message) {
    setMessages((prev) => [...prev, msg]);
    setTimeout(scrollToBottom, 0);
  }

  function scrollToBottom() {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }

  return (
    <div className="chat-shell">
      <header className="chat-header">
        <div>
          <h1>AI Summary</h1>
          <p className="subtitle">로컬 백엔드와 연동되는 ChatGPT 스타일 UI</p>
        </div>
        <div className="status">{loading ? "⏳ 요청 중..." : error ? "⚠️ 오류" : "준비 완료"}</div>
      </header>

      <div className="chat-body" ref={listRef}>
        {messages.map((m, idx) => (
          <div key={idx} className={`message ${m.role}`}>
            <div className="badge">{m.role === "user" ? "나" : m.role === "assistant" ? "비서" : "시스템"}</div>
            <div className="bubble">
              <p>{m.text}</p>
              {m.hits && m.hits.length > 0 && (
                <div className="hits">
                  <div className="hits-title">관련 문서</div>
                  <ul>
                    {m.hits.slice(0, 5).map((hit, i) => (
                      <li key={`${hit.path}-${i}`}>
                        <div className="hit-path">{hit.path}</div>
                        <div className="hit-meta">유사도: {hit.similarity?.toFixed(3) ?? "-"}</div>
                        {hit.preview && <div className="hit-preview">{hit.preview}</div>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <footer className="chat-input">
        <input
          value={input}
          placeholder="대화 입력… 검색/요약은 /search 로 시작"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!loading) send();
            }
          }}
          disabled={loading}
        />
        <button onClick={send} disabled={loading || !input.trim()}>
          전송
        </button>
      </footer>

      {error && <div className="error-box">⚠️ {error}</div>}
    </div>
  );
}
