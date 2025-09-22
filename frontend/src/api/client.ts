export interface SearchRequest {
  query: string;
  top_k?: number;
  session_id?: string | null;
}

export interface FeedbackRequest {
  session_id?: string | null;
  doc_id?: number;
  path?: string;
  ext?: string;
  owner?: string;
  action: "click" | "pin" | "like" | "dislike";
}

export interface SearchHit {
  path: string;
  ext: string;
  vector_similarity?: number;
  combined_score?: number;
  preview?: string;
  match_reasons?: string[];
  metadata_matches?: string[];
  score_breakdown?: Record<string, number>;
}

const API_BASE = "/api";

export async function search(req: SearchRequest) {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`search failed: ${res.status}`);
  }
  return res.json();
}

export async function sendFeedback(req: FeedbackRequest) {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`feedback failed: ${res.status}`);
  }
  return res.json();
}

export async function resetSession(sessionId?: string | null) {
  const res = await fetch(`${API_BASE}/session/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    throw new Error(`reset failed: ${res.status}`);
  }
  return res.json();
}

export async function triggerReindex(force = false) {
  const res = await fetch(`${API_BASE}/reindex`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
  if (!res.ok) {
    throw new Error(`reindex failed: ${res.status}`);
  }
  return res.json();
}
