import React, { useState } from "react";
import { ResultCard } from "../components/ResultCard";
import {
  search,
  sendFeedback,
  resetSession,
  triggerReindex,
  type SearchHit,
} from "../api/client";

export const SearchPage: React.FC = () => {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionHistory, setSessionHistory] = useState<string[]>([]);
  const [sessionPrefs, setSessionPrefs] = useState<{ exts: string[]; owners: string[] }>(
    { exts: [], owners: [] }
  );
  const [results, setResults] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await search({ query, session_id: sessionId ?? undefined });
      setResults(res.results ?? []);
      if (res.session_id) {
        setSessionId(res.session_id);
      }
      if (res.session?.history) {
        setSessionHistory(res.session.history);
      }
      if (res.session?.preferences) {
        setSessionPrefs(res.session.preferences);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const send = async (hit: SearchHit, action: "click" | "pin" | "like" | "dislike") => {
    try {
      await sendFeedback({
        session_id: sessionId ?? undefined,
        doc_id: (hit as any).doc_id,
        path: hit.path,
        ext: hit.ext,
        owner: (hit as any).owner,
        action,
      });
    } catch (err) {
      console.warn("feedback failed", err);
    }
  };

  const handleOpen = (hit: SearchHit) => send(hit, "click");
  const handleOpenFolder = (hit: SearchHit) => send(hit, "click");
  const handlePin = (hit: SearchHit) => send(hit, "pin");
  const handleLike = (hit: SearchHit) => send(hit, "like");
  const handleDislike = (hit: SearchHit) => send(hit, "dislike");

  const handleReset = async () => {
    const res = await resetSession(sessionId ?? undefined);
    setSessionId(res.session_id ?? null);
    setSessionHistory(res.history ?? []);
    setSessionPrefs({ exts: [], owners: [] });
    setResults([]);
  };

  const handleReindex = async () => {
    await triggerReindex(false);
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-4">
      <header className="space-y-2">
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 border rounded px-3 py-2"
            placeholder="검색어를 입력하세요"
          />
          <button className="btn" onClick={handleSearch} disabled={loading}>
            {loading ? "검색 중…" : "검색"}
          </button>
        </div>
        <div className="flex gap-2 text-sm">
          <button className="btn-secondary" onClick={handleReset}>
            세션 초기화
          </button>
          <button className="btn-secondary" onClick={handleReindex}>
            재색인 요청
          </button>
        </div>
        {sessionId && (
          <div className="text-xs text-gray-500">세션 ID: {sessionId}</div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-500">
          {sessionHistory.length > 0 && (
            <div>
              <div className="font-semibold">최근 질의</div>
              <ul className="space-y-1">
                {sessionHistory.slice(-5).map((item, idx) => (
                  <li key={idx} className="truncate" title={item}>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {(sessionPrefs.exts.length > 0 || sessionPrefs.owners.length > 0) && (
            <div className="space-y-1">
              {sessionPrefs.exts.length > 0 && (
                <div>
                  <div className="font-semibold">선호 확장자</div>
                  <div className="flex flex-wrap gap-1">
                    {sessionPrefs.exts.map((ext) => (
                      <span key={ext} className="badge">
                        {ext}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {sessionPrefs.owners.length > 0 && (
                <div>
                  <div className="font-semibold">선호 작성자</div>
                  <div className="flex flex-wrap gap-1">
                    {sessionPrefs.owners.map((owner) => (
                      <span key={owner} className="badge">
                        {owner}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </header>
      <main className="grid gap-3">
        {results.map((hit, idx) => (
          <ResultCard
            key={`${hit.path}-${idx}`}
            hit={hit}
            index={idx}
            onOpen={handleOpen}
            onOpenFolder={handleOpenFolder}
            onPin={handlePin}
            onLike={handleLike}
            onDislike={handleDislike}
          />
        ))}
        {!loading && results.length === 0 && (
          <div className="text-center text-gray-400 text-sm">결과가 없습니다.</div>
        )}
      </main>
    </div>
  );
};
