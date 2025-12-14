# pipeline.py  (Step2: 추출 + 학습)
import importlib
import subprocess
import json
import tempfile
import io
import math
import os
import platform
import re
import sys
import threading
import time
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, List, Tuple, Union, Set

from core.data_pipeline.custom_metadata import get_metadata_for_path
from core.data_pipeline.cache_manager import ChunkCache, SQLiteChunkCache
from core.data_pipeline.incremental import (
    load_scan_state,
    filter_incremental_rows,
    update_scan_state,
    save_scan_state,
)
from core.data_pipeline.embedder import AsyncSentenceEmbedder
from core.data_pipeline.evaluate import evaluate_embeddings

import numpy as np

# ---- 선택 의존성(있으면 사용) ----
try:
    import pandas as pd
except Exception:
    pd = None
PARQUET_ENGINE: Optional[str] = None
if pd is not None:
    for candidate in ("fastparquet", "pyarrow"):
        try:
            importlib.import_module(candidate)
            PARQUET_ENGINE = candidate
            break
        except ImportError:
            continue
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None
try:
    import docx
except Exception:
    docx = None
try:
    import pptx
except Exception:
    pptx = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None
try:
    import win32com.client
except Exception:
    win32com = None
try:
    import pythoncom
except Exception:
    pythoncom = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import joblib
except Exception:
    joblib = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.pipeline import Pipeline
    from sklearn import __version__ as sklearn_version
except Exception:
    TfidfVectorizer = TruncatedSVD = MiniBatchKMeans = Pipeline = None
    sklearn_version = "0"

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import olefile
except Exception:
    olefile = None
try:
    import pyhwp
except Exception:
    pyhwp = None


# =========================
# 콘솔 진행도 유틸
# =========================
class Spinner:
    FRAMES = ["|", "/", "-", "\\"]
    def __init__(self, prefix="", interval=0.12):
        self.prefix = prefix
        self.interval = interval
        self._stop = threading.Event()
        self._t = None
        self._i = 0
    def start(self) -> None:
        if self._t:
            return

        def _run() -> None:
            while not self._stop.wait(self.interval):
                frame = self.FRAMES[self._i % len(self.FRAMES)]
                self._i += 1
                sys.stdout.write(f"\r{self.prefix} {frame} ")
                sys.stdout.flush()

        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()

    def stop(self, clear=True) -> None:
        if not self._t:
            return
        self._stop.set()
        self._t.join()
        if clear:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

class ProgressLine:
    def __init__(self, total: int, label: str, update_every: int = 10):
        self.total = max(1, total)
        self.label = label
        self.update_every = max(1, update_every)
        self.start = time.time()
        self.n = 0

    def update(self, k: int = 1):
        self.n += k
        if (self.n % self.update_every) != 0 and self.n < self.total:
            return
        pct = min(100.0, self.n / self.total * 100.0)
        elapsed = time.time() - self.start
        rate = self.n/elapsed if elapsed>0 else 0
        remain = (self.total - self.n)/rate if rate>0 else 0
        sys.stdout.write(
            f"\r[{pct:5.1f}%] {self.label}  {self.n:,}/{self.total:,}  "
            f"{rate:,.1f}/s  elapsed={self._fmt(elapsed)}  ETA={self._fmt(remain)}   "
        )
        sys.stdout.flush()

    def close(self) -> None:
        self.n = self.total
        self.update(0)
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def _fmt(s: float) -> str:
        if s == float("inf"):
            return "∞"
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"


# =========================
# 텍스트 클린
# =========================
class TextCleaner:
    _multi = re.compile(r"\s+")

    @classmethod
    def clean(cls, s: str) -> str:
        if not s:
            return ""
        s = "".join(ch if ch.isprintable() or ch in "\t\n\r" else " " for ch in s)
        s = s.replace("\x00", " ")
        return cls._multi.sub(" ", s).strip()

TOKEN_PATTERN = r'(?u)(?:[가-힣]{1,}|[A-Za-z0-9]{2,})'

# 고정된 SVD 차원 수. Index/모델 불일치를 막기 위해 한곳에서 정의한다.
DEFAULT_N_COMPONENTS = 128
MODEL_TEXT_COLUMN = "text_model"
_META_SPLIT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
def _default_embed_model() -> str:
    env_model = os.getenv("DEFAULT_EMBED_MODEL")
    if env_model:
        return env_model
    if platform.system() == "Darwin":
        # Prefer the bundled multilingual-e5-small copy on macOS for stability
        return "models--intfloat--multilingual-e5-small"
    return "BAAI/bge-m3"


DEFAULT_EMBED_MODEL = _default_embed_model()
MODEL_TYPE_SENTENCE_TRANSFORMER = "sentence-transformer"

DEFAULT_CHUNK_MIN_TOKENS = 200
DEFAULT_CHUNK_MAX_TOKENS = 500

EMBED_DTYPE_ENV = "INFOPILOT_EMBED_DTYPE"
_VALID_EMBED_DTYPES = {"auto", "fp16", "fp32"}
CACHE_BACKEND_ENV = "INFOPILOT_CACHE_BACKEND"
_VALID_CACHE_BACKENDS = {"json", "sqlite"}


def _sanitize_embed_dtype(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in _VALID_EMBED_DTYPES else None


def _sanitize_cache_backend(value: Optional[str]) -> str:
    if not value:
        return "json"
    normalized = str(value).strip().lower()
    return normalized if normalized in _VALID_CACHE_BACKENDS else "json"


def _create_chunk_cache(path: Path) -> ChunkCache:
    backend = _sanitize_cache_backend(os.getenv(CACHE_BACKEND_ENV))
    actual_path = path
    if backend == "sqlite":
        if actual_path.suffix.lower() == ".json":
            actual_path = actual_path.with_suffix(".sqlite")
        elif not actual_path.name.endswith(".sqlite"):
            actual_path = actual_path.with_name(actual_path.name + ".sqlite")
    if backend == "sqlite":
        print(f"⚙️ Chunk cache: SQLite backend → {actual_path}", flush=True)
        return SQLiteChunkCache(actual_path)
    if backend != "json":
        print(f"⚠️ 지원하지 않는 캐시 백엔드 '{backend}' → json으로 대체합니다.", flush=True)
    return ChunkCache(path)

_TOKEN_REGEX = re.compile(TOKEN_PATTERN)


def _hash_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()

_EN_STOPWORDS: Set[str] = {
    "the", "and", "for", "that", "with", "from", "this", "have", "been", "were",
    "into", "about", "after", "before", "while", "shall", "could", "would", "there",
    "their", "which", "should", "among", "within", "between", "through", "without",
    "because", "against", "during", "under", "over", "where", "when", "whose", "them",
    "they", "these", "those", "ours", "your", "yours", "ourselves", "yourself",
    "yourselves", "myself", "been", "being", "also", "very", "much", "many", "such",
    "than", "ever", "here", "there", "once", "often", "again", "every", "across",
    "of", "in", "on", "at", "by", "is", "are", "be", "am", "was", "were", "it",
    "its", "as", "to", "or", "an", "a", "so", "if", "not", "no", "do", "does",
    "did", "each", "per", "via", "both", "same", "own", "due", "per", "via",
}

_KO_STOPWORDS: Set[str] = {
    "그리고", "그러나", "하지만", "그러면서", "그러므로", "또한", "그러니까", "따라서", "그리고나서",
    "그러면", "그리고도", "그러곤", "그러했지만", "그러할", "그러하다", "그러한", "그런", "이는",
    "이는", "이를", "있는", "있으며", "있습니다", "합니다", "하였다", "하는", "하게", "하고",
    "하여", "하여금", "해서", "하지만", "혹은", "또는", "부터", "까지", "위해", "대한", "다른",
    "모든", "각각", "관련", "경우", "때문", "때문에", "여러", "어떤", "일부", "특히", "다만",
    "즉", "따위", "예를", "예를들어", "수", "등", "및", "것", "그리고", "또", "또한",
    "우리", "너희", "그것", "이것", "저것", "그", "이", "저", "에게", "에서", "으로",
    "로", "에는", "에는", "였다", "이며", "면서", "이라", "이라서",
}

_DOMAIN_STOPWORDS: Set[str] = {
    "document",
    "documents",
    "report",
    "reports",
    "file",
    "files",
    "data",
    "자료",
    "파일",
    "문서",
    "보고서",
    "첨부",
    "자료들",
    "내용",
    "프로젝트",
    "관련자료",
}

_STOPWORDS: Set[str] = {
    word.lower() for word in (*_EN_STOPWORDS, *_KO_STOPWORDS, *_DOMAIN_STOPWORDS)
}



def _split_tokens(source: str) -> List[str]:
    if not source:
        return []
    return [tok for tok in _META_SPLIT_RE.split(source) if tok]


def _remove_stopwords(text: str) -> str:
    if not text:
        return ""
    kept: List[str] = []
    for match in _TOKEN_REGEX.finditer(text):
        token = match.group(0)
        token_norm = token.lower()
        if token_norm in _STOPWORDS:
            continue
        if token_norm.isdigit():
            continue
        if len(set(token_norm)) == 1 and len(token_norm) <= 3:
            continue
        kept.append(token)
    if not kept:
        return text.strip()
    return " ".join(kept)


def _slice_text_by_ratio(source: str, start_char: int, end_char: int, base_len: int) -> str:
    if not source:
        return ""
    if base_len <= 0:
        return source.strip()
    length = len(source)
    start_ratio = max(0.0, min(1.0, float(start_char) / float(base_len)))
    end_ratio = max(start_ratio, min(1.0, float(end_char) / float(base_len)))
    start_idx = int(round(start_ratio * length))
    end_idx = int(round(end_ratio * length))
    if end_idx <= start_idx:
        end_idx = min(length, max(start_idx + 1, end_idx))
    return source[start_idx:end_idx].strip()


def _token_chunk_spans(text: str, *, min_tokens: int, max_tokens: int) -> List[Tuple[int, int, int]]:
    if not text or not text.strip():
        cleaned = (text or "").strip()
        return [(0, len(text), 0)] if cleaned else []

    matches = list(_TOKEN_REGEX.finditer(text))
    total_tokens = len(matches)
    if total_tokens == 0:
        cleaned = text.strip()
        return [(0, len(text), 0)] if cleaned else []
    if total_tokens <= max_tokens:
        return [(0, len(text), total_tokens)]

    spans: List[Tuple[int, int, int]] = []
    start_index = 0
    prev_char = 0
    text_len = len(text)

    while start_index < total_tokens:
        end_index = min(start_index + max_tokens, total_tokens)
        remaining = total_tokens - end_index
        if remaining and remaining < min_tokens:
            end_index = total_tokens
        next_start_char = matches[end_index].start() if end_index < total_tokens else text_len
        span_start = prev_char
        span_end = next_start_char
        token_count = end_index - start_index
        chunk = text[span_start:span_end].strip()
        if chunk:
            spans.append((span_start, span_end, token_count))
        prev_char = next_start_char
        start_index = end_index

    if len(spans) >= 2 and spans[-1][2] < min_tokens:
        prev_start, _prev_end, prev_tokens = spans[-2]
        spans[-2] = (prev_start, spans[-1][1], prev_tokens + spans[-1][2])
        spans.pop()

    if spans and spans[-1][1] < text_len:
        start, _, tokens = spans[-1]
        spans[-1] = (start, text_len, tokens)

    return spans


def _token_chunk_spans_with_overlap(
    text: str,
    *,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[int, int, int]]:
    min_tokens = max(1, int(min_tokens))
    max_tokens = max(min_tokens, int(max_tokens))
    overlap = max(0, int(overlap_tokens))

    matches = list(_TOKEN_REGEX.finditer(text))
    if not matches:
        cleaned = (text or "").strip()
        return [(0, len(text), 0)] if cleaned else []

    total_tokens = len(matches)
    if total_tokens <= max_tokens:
        return [(0, len(text), total_tokens)]

    spans: List[Tuple[int, int, int]] = []
    start_index = 0
    text_len = len(text)
    while start_index < total_tokens:
        end_index = min(start_index + max_tokens, total_tokens)
        remaining = total_tokens - end_index
        if remaining and remaining < min_tokens:
            end_index = total_tokens

        start_char = matches[start_index].start()
        end_char = matches[end_index - 1].end() if end_index > start_index else min(text_len, start_char)
        chunk = text[start_char:end_char].strip()
        if chunk:
            spans.append((start_char, end_char, end_index - start_index))

        if end_index >= total_tokens:
            break
        next_start = end_index - overlap if overlap else end_index
        if next_start <= start_index:
            next_start = end_index
        start_index = next_start

    if spans and spans[-1][1] < text_len:
        start, _, tokens = spans[-1]
        spans[-1] = (start, text_len, tokens)

    return spans


_MD_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_MD_NUMBERED_HEADING_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*[\).])\s+(.+?)\s*$")


def _iter_markdown_sections(text: str) -> List[Tuple[int, int, str]]:
    """Return (start_char, end_char, heading_title) for markdown-ish sections."""
    if not text:
        return []
    headings: List[Tuple[int, str]] = []
    for match in _MD_HEADING_RE.finditer(text):
        title = (match.group(2) or "").strip()
        headings.append((match.start(), title))
    for match in _MD_NUMBERED_HEADING_RE.finditer(text):
        title = (match.group(2) or "").strip()
        headings.append((match.start(), title))
    if not headings:
        return [(0, len(text), "")]

    headings.sort(key=lambda item: item[0])
    sections: List[Tuple[int, int, str]] = []
    for idx, (start, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        sections.append((start, end, title))
    if sections and sections[0][0] > 0:
        sections.insert(0, (0, sections[0][0], ""))
    return [(s, e, t) for (s, e, t) in sections if s < e]


def _is_markdown_record(record: Dict[str, Any]) -> bool:
    ext = str(record.get("ext") or "").lower()
    if ext == ".md":
        return True
    meta = record.get("meta")
    if isinstance(meta, dict) and str(meta.get("format") or "").lower() == "markdown":
        return True
    return False


def _adaptive_chunk_window(text: str, base_min: int, base_max: int) -> Tuple[int, int]:
    base_min = max(16, int(base_min))
    base_max = max(base_min + 16, int(base_max))
    approx_tokens = len(_TOKEN_REGEX.findall(text)) or max(1, len(text) // 4)

    if approx_tokens <= base_max:
        min_tokens = max(16, int(base_min * 0.5))
        max_tokens = max(min_tokens + 24, int(base_max * 0.75))
        return min_tokens, max_tokens

    if approx_tokens <= base_max * 3:
        return base_min, base_max

    scale = min(2.0, approx_tokens / float(base_max * 3))
    min_tokens = int(base_min * (1.0 + (scale * 0.5)))
    max_tokens = int(base_max * (1.0 + (scale * 0.5)))
    min_tokens = max(base_min, min(min_tokens, 320))
    max_tokens = max(min_tokens + 32, min(1200, max_tokens))
    remainder = approx_tokens - max_tokens
    if remainder and remainder < min_tokens:
        adjustment = min_tokens - remainder
        max_tokens = max(min_tokens + 32, max_tokens - adjustment)
    return min_tokens, max_tokens


def _apply_uniform_chunks(
    df: "pd.DataFrame",
    *,
    min_tokens: int = DEFAULT_CHUNK_MIN_TOKENS,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    overlap_tokens: int = 0,
) -> "pd.DataFrame":
    if pd is None or df is None or df.empty or "text" not in df.columns:
        return df

    records = df.to_dict(orient="records")
    chunked: List[Dict[str, Any]] = []

    for record in records:
        base_text = str(record.get("text") or "")
        adaptive_min, adaptive_max = _adaptive_chunk_window(base_text, min_tokens, max_tokens)
        original_text = record.get("text_original") or ""
        base_len = len(base_text)

        spans: List[Tuple[int, int, int]] = []
        headings: List[str] = []
        if _is_markdown_record(record):
            overlap = max(0, int(overlap_tokens or 30))
            for section_start, section_end, heading in _iter_markdown_sections(base_text):
                section_text = base_text[section_start:section_end]
                for start_char, end_char, token_count in _token_chunk_spans_with_overlap(
                    section_text,
                    min_tokens=adaptive_min,
                    max_tokens=adaptive_max,
                    overlap_tokens=overlap,
                ):
                    spans.append((section_start + start_char, section_start + end_char, token_count))
                    headings.append(heading)
        else:
            spans = _token_chunk_spans(base_text, min_tokens=adaptive_min, max_tokens=adaptive_max)
            headings = ["" for _ in spans]
        if not spans:
            new_rec = dict(record)
            new_rec["chunk_id"] = 1
            new_rec["chunk_count"] = 1
            new_rec["chunk_tokens"] = 0
            preview_source = record.get("text_original") or record.get("text") or ""
            new_rec["text"] = _remove_stopwords(base_text)
            new_rec["text_original"] = preview_source
            new_rec["preview"] = str(preview_source).strip()[:360]
            new_rec["doc_hash"] = record.get("doc_hash", "")
            new_rec["content_hash"] = _hash_text(new_rec["text"])
            chunked.append(new_rec)
            continue

        chunk_count = max(1, len(spans))

        for idx, (start_char, end_char, token_count) in enumerate(spans, start=1):
            chunk_slice = base_text[start_char:end_char].strip()
            filtered_chunk = _remove_stopwords(chunk_slice)
            if not filtered_chunk:
                filtered_chunk = chunk_slice

            new_rec = dict(record)
            new_rec["chunk_id"] = idx
            new_rec["chunk_count"] = chunk_count
            new_rec["chunk_tokens"] = token_count
            new_rec["text"] = filtered_chunk
            if headings and idx - 1 < len(headings) and headings[idx - 1]:
                new_rec["heading"] = headings[idx - 1]

            if isinstance(original_text, str) and original_text:
                orig_chunk = _slice_text_by_ratio(original_text, start_char, end_char, base_len)
            else:
                orig_chunk = chunk_slice
            new_rec["text_original"] = orig_chunk
            new_rec["preview"] = (orig_chunk or chunk_slice).strip()[:360]
            new_rec["doc_hash"] = record.get("doc_hash", "")
            new_rec["content_hash"] = _hash_text(filtered_chunk or chunk_slice)

            chunked.append(new_rec)

    return pd.DataFrame(chunked)


def _time_tokens(epoch: Optional[float]) -> List[str]:
    if not epoch:
        return []
    try:
        dt = datetime.fromtimestamp(float(epoch))
    except Exception:
        return []
    parts = [
        dt.strftime("%Y"),
        dt.strftime("%Y-%m"),
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%B"),
        dt.strftime("%m"),
    ]
    return parts


def _size_bucket(size: Optional[int]) -> Optional[str]:
    if size is None:
        return None
    try:
        size = int(size)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    if size < 10 * 1024:
        return "size:tiny"
    if size < 1 * 1024 * 1024:
        return "size:small"
    if size < 10 * 1024 * 1024:
        return "size:medium"
    if size < 50 * 1024 * 1024:
        return "size:large"
    return "size:huge"


def _metadata_text(
    path: str,
    ext: str,
    drive: str,
    size: Optional[int] = None,
    mtime: Optional[float] = None,
    ctime: Optional[float] = None,
    owner: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    tokens: List[str] = []
    extra_clean: Optional[str] = None
    if path:
        try:
            p = Path(path)
        except Exception:
            p = None
        if p:
            name = p.name
            if name:
                tokens.append(name)
            stem = p.stem
            if stem and stem != name:
                tokens.append(stem)
        else:
            tokens.append(str(path))
    if ext:
        ext_clean = str(ext).strip()
        if ext_clean:
            tokens.append(ext_clean)
            ext_no_dot = ext_clean.lstrip(".")
            if ext_no_dot:
                tokens.append(ext_no_dot)
    if drive:
        drive_str = str(drive)
        tokens.append(drive_str)
    for epoch in (mtime, ctime):
        tokens.extend(_time_tokens(epoch))
    bucket = _size_bucket(size)
    if bucket:
        tokens.append(bucket)
    if owner:
        tokens.append(str(owner))
    if extra:
        extra_clean = TextCleaner.clean(str(extra))
        if extra_clean:
            tokens.extend(_split_tokens(extra_clean))

    seen = set()
    normalized: List[str] = []
    for token in tokens:
        cleaned = TextCleaner.clean(str(token)).lower()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    metadata_text = " ".join(normalized)
    if extra_clean:
        return f"{metadata_text}\n{extra_clean}" if metadata_text else extra_clean
    return metadata_text


def _compose_model_text(base_text: str, metadata: str) -> str:
    base = (base_text or "").strip()
    meta = (metadata or "").strip()
    if base:
        if meta and len(base) < 40:
            return f"{base}\n\n{meta}"
        return base
    return meta


_DOC_TAG_HINTS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "law",
        (
            "법령",
            "법률",
            "법규",
            "조례",
            "규정",
            "규칙",
            "지침",
            "세칙",
            "훈령",
            "행정규칙",
            "regulation",
            "ordinance",
            "bylaw",
        ),
        ("법령문서", "법규", "조례"),
    ),
    (
        "notice",
        (
            "공고",
            "공지",
            "고시",
            "입찰",
            "계약",
            "발주",
            "제안요청서",
            "rfp",
            "announcement",
            "notice",
        ),
        ("공고문", "입찰", "공지문"),
    ),
    (
        "report",
        (
            "보고서",
            "분석",
            "결과보고",
            "백서",
            "리포트",
            "report",
            "analysis",
        ),
        ("보고서", "분석자료"),
    ),
    (
        "minutes",
        (
            "회의록",
            "의사록",
            "minutes",
            "meeting minutes",
        ),
        ("회의록", "회의기록"),
    ),
    (
        "plan",
        (
            "계획",
            "전략",
            "로드맵",
            "마스터플랜",
            "plan",
            "strategy",
            "roadmap",
        ),
        ("계획서", "전략문서"),
    ),
    (
        "manual",
        (
            "매뉴얼",
            "지침서",
            "가이드",
            "guide",
            "manual",
        ),
        ("지침서", "가이드"),
    ),
)


def _infer_doc_tags(path_value: str, extra: Optional[str]) -> Tuple[List[str], List[str]]:
    haystack_parts = []
    if path_value:
        haystack_parts.append(str(path_value))
        try:
            path_obj = Path(path_value)
            haystack_parts.append(path_obj.name)
            haystack_parts.append(path_obj.stem)
        except Exception:
            pass
    if extra:
        haystack_parts.append(str(extra))
    haystack = " ".join(part for part in haystack_parts if part).lower()
    if not haystack:
        return [], []
    tags: List[str] = []
    tokens: List[str] = []
    for slug, keywords, tag_tokens in _DOC_TAG_HINTS:
        for keyword in keywords:
            if keyword.lower() in haystack:
                tags.append(slug)
                for token in tag_tokens:
                    cleaned = TextCleaner.clean(token)
                    if cleaned:
                        tokens.append(cleaned)
                break
    if not tags:
        return [], []
    # deduplicate while preserving order
    seen_tags = set()
    ordered_tags: List[str] = []
    for tag in tags:
        if tag not in seen_tags:
            seen_tags.add(tag)
            ordered_tags.append(tag)
    seen_tokens = set()
    ordered_tokens: List[str] = []
    for token in tokens:
        if token not in seen_tokens:
            seen_tokens.add(token)
            ordered_tokens.append(token)
    return ordered_tags, ordered_tokens


def _prepare_text_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    if pd is None or df is None:
        return df
    if df.empty:
        if MODEL_TEXT_COLUMN not in df.columns:
            df[MODEL_TEXT_COLUMN] = pd.Series(dtype=str)
        return df

    for column in ("text", "text_original"):
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    if "text" not in df.columns:
        df["text"] = ""

    paths = df.get("path")
    if paths is None:
        paths = pd.Series([""] * len(df))
    else:
        paths = paths.fillna("").astype(str)

    exts = df.get("ext")
    if exts is None:
        exts = pd.Series([""] * len(df))
    else:
        exts = exts.fillna("").astype(str)

    drives = df.get("drive")
    if drives is None:
        drives = pd.Series([""] * len(df))
    else:
        drives = drives.fillna("").astype(str)

    sizes = df.get("size")
    if sizes is None:
        sizes = pd.Series([0] * len(df))
    else:
        sizes = sizes.fillna(0)

    mtimes = df.get("mtime")
    if mtimes is None:
        mtimes = pd.Series([0.0] * len(df))
    else:
        mtimes = mtimes.fillna(0.0)

    ctimes = df.get("ctime")
    if ctimes is None:
        ctimes = pd.Series([0.0] * len(df))
    else:
        ctimes = ctimes.fillna(0.0)

    owners = df.get("owner")
    if owners is None:
        owners = pd.Series([""] * len(df))
    else:
        owners = owners.fillna("").astype(str)

    base_texts = df["text"].tolist()
    extra_texts = [
        get_metadata_for_path(str(paths.iat[idx]))
        for idx in range(len(df))
    ]
    metadata_list: List[str] = []
    doc_tags: List[List[str]] = []
    doc_primary_tags: List[str] = []
    for idx in range(len(df)):
        tags, tag_tokens = _infer_doc_tags(paths.iat[idx], extra_texts[idx])
        doc_tags.append(tags)
        doc_primary_tags.append(tags[0] if tags else "")
        metadata_value = _metadata_text(
            paths.iat[idx],
            exts.iat[idx],
            drives.iat[idx],
            size=sizes.iat[idx],
            mtime=mtimes.iat[idx],
            ctime=ctimes.iat[idx],
            owner=owners.iat[idx],
            extra=extra_texts[idx],
        )
        if tag_tokens:
            tag_text = " ".join(tag_tokens)
            metadata_value = f"{metadata_value} {tag_text}".strip() if metadata_value else tag_text
        metadata_list.append(metadata_value)
    df["doc_tags"] = doc_tags
    df["doc_primary_tag"] = doc_primary_tags
    df[MODEL_TEXT_COLUMN] = [
        _compose_model_text(base_texts[idx], metadata_list[idx])
        for idx in range(len(df))
    ]
    return df


def _deduplicate_corpus(df: "pd.DataFrame") -> "pd.DataFrame":
    """Remove duplicate chunks based on content hash (and chunk_id when available)."""
    if pd is None or df is None or df.empty:
        return df
    if "content_hash" not in df.columns or "path" not in df.columns:
        return df

    working = df.copy()
    working["_dedup_pref"] = working["path"].apply(
        lambda p: 1 if str(p or "").startswith("/Volumes/") else 0
    )
    working["_dedup_len"] = working["path"].apply(lambda p: len(str(p or "")))
    working["_dedup_order"] = range(len(working))

    subset_cols = ["content_hash"]
    if "chunk_id" in working.columns:
        subset_cols.append("chunk_id")

    working = working.sort_values(subset_cols + ["_dedup_pref", "_dedup_len", "_dedup_order"])
    working = working.drop_duplicates(subset=subset_cols, keep="first")
    working = working.sort_values("_dedup_order").drop(columns=["_dedup_pref", "_dedup_len", "_dedup_order"])
    working = working.reset_index(drop=True)
    return working


def _resolve_kmeans_n_init() -> Union[str, int]:
    """Return MiniBatchKMeans n_init compatible with installed scikit-learn."""
    try:
        parts = (sklearn_version or "0").split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        if (major, minor) >= (1, 4):
            return "auto"
    except Exception:
        pass
    return 3


# =========================
# Extractors
# =========================
class BaseExtractor:
    exts: Tuple[str, ...] = ()

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.exts

    def extract(self, path: Path) -> Dict[str, Any]:
        raise NotImplementedError

class HwpExtractor(BaseExtractor):
    exts = (".hwp",)

    def extract(self, p: Path) -> Dict[str, Any]:
        system = platform.system().lower()
        if system.startswith("win") and win32com:
            com_initialized = False
            try:
                if pythoncom:
                    pythoncom.CoInitialize()
                    com_initialized = True
                app = win32com.Dispatch("HWPFrame.HwpObject")
                try:
                    app.Open(str(p))
                    text = app.GetTextFile("TEXT", "") or ""
                    return {
                        "ok": True,
                        "text": TextCleaner.clean(text),
                        "meta": {"engine": "win32com-hwp"},
                    }
                finally:
                    try:
                        app.Quit()
                    except Exception:
                        pass
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"HWP win32com 실패: {exc}"}}
            finally:
                if com_initialized and pythoncom:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
        if olefile and pyhwp:
            try:
                from pyhwp.hwp5txt import hwp5txt  # type: ignore

                with olefile.OleFileIO(str(p)) as ole:
                    buf = io.StringIO()
                    hwp5txt(ole, buf)
                    text = buf.getvalue()
                cleaned = TextCleaner.clean(text)
                if cleaned:
                    return {
                        "ok": True,
                        "text": cleaned,
                        "meta": {"engine": "pyhwp", "bytes": p.stat().st_size},
                    }
            except Exception as exc:
                return {
                    "ok": False,
                    "text": "",
                    "meta": {"error": f"HWP pyhwp 추출 실패: {exc}"},
                }
        return {
            "ok": False,
            "text": "",
            "meta": {"error": "HWP 추출을 위해서는 Windows + 한/글 환경이 필요합니다."},
        }


class DocDocxExtractor(BaseExtractor):
    exts = (".doc", ".docx")

    def extract(self, p: Path) -> Dict[str, Any]:
        suffix = p.suffix.lower()
        if suffix == ".docx" and docx:
            try:
                document = docx.Document(str(p))
                text = "\n".join(par.text for par in document.paragraphs)
                return {
                    "ok": True,
                    "text": TextCleaner.clean(text),
                    "meta": {"engine": "python-docx", "paras": len(document.paragraphs)},
                }
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"DOCX parse failed: {exc}"}}

        system = platform.system().lower()
        if suffix == ".doc" and system.startswith("win") and win32com:
            com_initialized = False
            try:
                if pythoncom:
                    pythoncom.CoInitialize()
                    com_initialized = True
                word = win32com.Dispatch("Word.Application")
                word.Visible = False
                try:
                    doc_obj = word.Documents.Open(str(p), ReadOnly=True)
                    try:
                        text = doc_obj.Content.Text or ""
                    finally:
                        doc_obj.Close(False)
                finally:
                    try:
                        word.Quit()
                    except Exception:
                        pass
                return {
                    "ok": True,
                    "text": TextCleaner.clean(text),
                    "meta": {"engine": "win32com-word"},
                }
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"DOC win32com 실패: {exc}"}}
            finally:
                if com_initialized and pythoncom:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

        return {
            "ok": False,
            "text": "",
            "meta": {"error": "DOC/DOCX 추출을 위해 python-docx 또는 Windows Word가 필요합니다."},
        }

class ExcelLikeExtractor(BaseExtractor):
    exts=(".xlsx",".xls",".xlsm",".xlsb",".xltx",".csv")
    def extract(self, p:Path)->Dict[str,Any]:
        if pd is None:
            return {"ok":False,"text":"","meta":{"error":"pandas required"}}
        try:
            if p.suffix.lower()==".csv":
                df=pd.read_csv(p, nrows=200, encoding="utf-8", engine="python")
                txt=self._df_to_text(df)
                return {"ok":True,"text":txt,"meta":{"engine":"pandas","columns":df.columns.tolist(), "rows_preview":min(200,len(df))}}
            eng = "openpyxl" if p.suffix.lower() in (".xlsx",".xlsm",".xltx") else ("xlrd" if p.suffix.lower()==".xls" else "pyxlsb")
            sheets = pd.read_excel(p, sheet_name=None, nrows=200, engine=eng)
            parts=[]
            for s,df_sheet in sheets.items():
                parts.append(f"[Sheet:{s}]")
                parts.append(" | ".join(map(str, df_sheet.columns.tolist())))
                for _,row in df_sheet.head(50).iterrows():
                    parts.append(" • "+" | ".join(map(lambda x: str(x), row.tolist())))
            return {"ok":True,"text":TextCleaner.clean("\n".join(parts)),"meta":{"engine":"pandas","sheets":list(sheets.keys())}}
        except Exception as e:
            detail = str(e)
            if "openpyxl" in detail.lower():
                detail += " (pip install openpyxl)"
            return {"ok":False,"text":"","meta":{"error":f"excel/csv read failed: {detail}"}}
    @staticmethod
    def _df_to_text(df)->str:
        cols=" | ".join(map(str, df.columns.tolist()))
        rows=[]
        for _,row in df.head(50).iterrows():
            rows.append(" • "+" | ".join(map(lambda x: str(x), row.tolist())))
        return TextCleaner.clean(f"{cols}\n"+"\n".join(rows))

class PdfExtractor(BaseExtractor):
    exts = (".pdf",)

    def extract(self, p: Path) -> Dict[str, Any]:
        if fitz:
            try:
                with fitz.open(str(p)) as doc:
                    page_count = doc.page_count
                    text = "\n".join(page.get_text("text") for page in doc)
                return {
                    "ok": True,
                    "text": TextCleaner.clean(text),
                    "meta": {"engine": "pymupdf", "pages": page_count},
                }
            except Exception:
                pass
        if pdfplumber:
            try:
                with pdfplumber.open(str(p)) as doc:
                    pages = [page.extract_text() or "" for page in doc.pages]
                text = "\n".join(pages)
                cleaned = TextCleaner.clean(text)
                if cleaned:
                    return {
                        "ok": True,
                        "text": cleaned,
                        "meta": {"engine": "pdfplumber", "pages": len(pages)},
                    }
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"PDF pdfplumber 실패: {exc}"}}
        if pdfminer_extract_text:
            try:
                text = pdfminer_extract_text(str(p))
                return {"ok": True, "text": TextCleaner.clean(text), "meta": {"engine": "pdfminer"}}
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"PDF pdfminer 실패: {exc}"}}
        return {"ok": False, "text": "", "meta": {"error": "PDF 추출 엔진이 설치되지 않았습니다."}}


class PptExtractor(BaseExtractor):
    exts = (".ppt", ".pptx")

    def extract(self, p: Path) -> Dict[str, Any]:
        suffix = p.suffix.lower()
        if suffix == ".pptx" and pptx:
            try:
                presentation = pptx.Presentation(str(p))
                texts: List[str] = []
                for idx, slide in enumerate(presentation.slides, 1):
                    parts: List[str] = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text = (shape.text or "").strip()
                            if text:
                                parts.append(text)
                    if parts:
                        texts.append(f"[Slide {idx}] " + " ".join(parts))
                return {
                    "ok": True,
                    "text": TextCleaner.clean("\n".join(texts)),
                    "meta": {"engine": "python-pptx", "slides": len(presentation.slides)},
                }
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"PPTX parse failed: {exc}"}}

        system = platform.system().lower()
        if suffix == ".ppt" and system.startswith("win") and win32com:
            com_initialized = False
            try:
                if pythoncom:
                    pythoncom.CoInitialize()
                    com_initialized = True
                powerpoint = win32com.Dispatch("PowerPoint.Application")
                powerpoint.Visible = False
                presentation = powerpoint.Presentations.Open(str(p), WithWindow=False)
                texts: List[str] = []
                try:
                    for slide in presentation.Slides:
                        parts = []
                        for shape in slide.Shapes:
                            has_text = hasattr(shape, "HasTextFrame") and shape.HasTextFrame
                            if has_text and shape.TextFrame.HasText:
                                parts.append(shape.TextFrame.TextRange.Text)
                        if parts:
                            texts.append(" ".join(parts))
                    return {
                        "ok": True,
                        "text": TextCleaner.clean("\n".join(texts)),
                        "meta": {"engine": "win32com-ppt"},
                    }
                finally:
                    presentation.Close()
                    powerpoint.Quit()
            except Exception as exc:
                return {"ok": False, "text": "", "meta": {"error": f"PPT win32com 실패: {exc}"}}
            finally:
                if com_initialized and pythoncom:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

        return {"ok": False, "text": "", "meta": {"error": "PPT/PPTX 추출을 위해 python-pptx 또는 Windows PowerPoint가 필요합니다."}}


class PlainTextExtractor(BaseExtractor):
    exts = (".txt", ".md", ".rst", ".log")

    def extract(self, p: Path) -> Dict[str, Any]:
        try:
            raw_text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"ok": False, "text": "", "meta": {"error": f"텍스트 파일 읽기 실패: {exc}"}}
        cleaned = TextCleaner.clean(raw_text)
        meta: Dict[str, Any] = {"engine": "plain-text"}
        if p.suffix.lower() == ".md":
            meta["format"] = "markdown"
        return {
            "ok": bool(cleaned),
            "text": cleaned,
            "text_original": raw_text,
            "meta": meta,
        }


class CodeExtractor(BaseExtractor):
    exts = (
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".sh",
        ".bash",
    )

    def extract(self, p: Path) -> Dict[str, Any]:
        try:
            raw_text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"ok": False, "text": "", "meta": {"error": f"코드 파일 읽기 실패: {exc}"}}
        cleaned = TextCleaner.clean(raw_text)
        meta: Dict[str, Any] = {"engine": "code", "extension": p.suffix.lower()}
        return {
            "ok": bool(cleaned),
            "text": cleaned,
            "text_original": raw_text,
            "meta": meta,
        }


EXTRACTORS = [
    HwpExtractor(),
    DocDocxExtractor(),
    ExcelLikeExtractor(),
    PdfExtractor(),
    PptExtractor(),
    PlainTextExtractor(),
    CodeExtractor(),
]
EXT_MAP={e:ex for ex in EXTRACTORS for e in ex.exts}


# =========================
# 코퍼스 빌더 (번역 기능 수정)
# =========================
@dataclass
class ExtractRecord:
    path: str
    ext: str
    ok: bool
    text: str
    text_original: str
    meta: Dict[str, Any]
    size: Optional[int] = None
    mtime: Optional[float] = None
    ctime: Optional[float] = None
    owner: Optional[str] = None
    doc_hash: str = ""
    file_hash: str = ""

class CorpusBuilder:
    MAX_TRANSLATE_CHARS = 4000

    def __init__(
        self,
        max_text_chars: int = 200_000,
        progress: bool = True,
        translate: bool = False,
        max_workers: Optional[int] = None,
        target_embed_dtype: str = "auto",
    ):
        self.max_text_chars = max_text_chars
        self.progress = progress
        self.translate = translate
        self.target_embed_dtype = _sanitize_embed_dtype(target_embed_dtype) or "auto"
        self.translator = None
        if translate:
            if GoogleTranslator is None:
                print("⚠️ 경고: 'deep-translator' 라이브러리를 찾을 수 없어 번역 기능이 비활성화됩니다.")
                print("   해결: pip install deep-translator")
            else:
                try:
                    self.translator = GoogleTranslator(source="auto", target="en")
                except Exception as exc:
                    print("⚠️ 경고: 번역기 초기화에 실패해 번역 기능이 비활성화됩니다.")
                    print(f"   상세: {exc}")
        worker_default = max(1, min(8, (os.cpu_count() or 4)))
        self.max_workers = max_workers or worker_default
        if self.translate:
            # 번역 시 외부 API 호출이 순차 처리되도록 워커 1개만 사용
            self.max_workers = 1

    def build(self, file_rows: List[Dict[str, Any]]):
        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")

        total = len(file_rows)
        if total == 0:
            print("ℹ️ 신규/변경 문서가 없어 추출을 건너뜁니다.", flush=True)
            empty = pd.DataFrame(columns=list(ExtractRecord.__annotations__.keys()))
            empty.attrs["target_embed_dtype"] = self.target_embed_dtype
            return empty

        use_tqdm = self.progress and tqdm is not None
        desc = "📥 Extract & Translate" if self.translate else "📥 Extract"
        bar = tqdm(total=total, desc=desc, unit="file") if use_tqdm else ProgressLine(total, "extracting", update_every=max(1, total // 100 or 1))

        recs: List[Optional[ExtractRecord]] = [None] * total
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            future_map = {
                executor.submit(self._extract_one, file_rows[idx]): idx
                for idx in range(total)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    rec = future.result()
                except Exception as exc:
                    row = file_rows[idx]
                    rec = ExtractRecord(
                        path=row.get("path", ""),
                        ext=row.get("ext", ""),
                        ok=False,
                        text="",
                        text_original="",
                        meta={"error": f"extract crash: {exc}"},
                        size=row.get("size"),
                        mtime=row.get("mtime"),
                        ctime=row.get("ctime"),
                        owner=row.get("owner"),
                    )
                recs[idx] = rec
                if use_tqdm:
                    bar.update(1)
                else:
                    bar.update(1)

        if use_tqdm and bar is not None:
            bar.close()
        elif not use_tqdm:
            bar.close()

        records = [r.__dict__ for r in recs if r is not None]
        df = pd.DataFrame(records)
        df.attrs["target_embed_dtype"] = self.target_embed_dtype
        _prepare_text_frame(df)
        ok = int(df["ok"].sum()) if len(df) > 0 else 0
        fail = int((~df["ok"]).sum()) if len(df) > 0 else 0
        print(f"✅ Extract 완료: ok={ok}, fail={fail}", flush=True)
        return df

    def _extract_one(self, row: Dict[str, Any]) -> ExtractRecord:
        path = Path(row["path"])
        ext = path.suffix.lower()
        file_hash = str(row.get("hash") or row.get("file_hash") or "").strip()
        mask_pii = bool(row.get("policy_mask_pii") or row.get("mask_pii"))
        ex = EXT_MAP.get(ext)
        if not ex:
            return ExtractRecord(
                str(path),
                ext,
                False,
                "",
                "",
                {"error": "no extractor"},
                row.get("size"),
                row.get("mtime"),
                row.get("ctime"),
                row.get("owner"),
                "",
                file_hash,
            )
        try:
            out = ex.extract(path)
            raw_text = (out.get("text", "") or "")[:self.max_text_chars]
            doc_hash = _hash_text(raw_text)

            if mask_pii and raw_text.strip():
                # Reuse meeting pipeline masking rules for privacy-preserving corpora.
                from core.agents.meeting.pii import mask_text as _mask_text

                original_text = _mask_text(raw_text)
            else:
                original_text = raw_text

            text_for_model = original_text
            if self.translator and original_text.strip():
                text_for_model = self._translate_text(original_text, context=path.name)

            return ExtractRecord(
                str(path),
                ext,
                bool(out.get("ok", False)),
                text_for_model,
                original_text,
                out.get("meta", {}),
                row.get("size"),
                row.get("mtime"),
                row.get("ctime"),
                row.get("owner"),
                doc_hash,
                file_hash,
            )
        except Exception as e:
            return ExtractRecord(
                str(path),
                ext,
                False,
                "",
                "",
                {"error": f"extract crash: {e}"},
                row.get("size"),
                row.get("mtime"),
                row.get("ctime"),
                row.get("owner"),
                "",
                file_hash,
            )

    def _translate_text(self, text: str, *, context: str) -> str:
        if not self.translator:
            return text
        chunks = self._chunk_text(text, self.MAX_TRANSLATE_CHARS)
        try:
            translated_chunks: List[str] = []
            for chunk in chunks:
                translated = self.translator.translate(chunk)
                translated_chunks.append(self._translated_text(translated, fallback=chunk))
            joined = "\n".join(translated_chunks).strip()
            return joined or text
        except Exception as exc:
            self._log_warning(f"\n[경고] '{context}' 번역 실패. 원본 텍스트 사용. 오류: {exc}")
            return text

    @staticmethod
    def _translated_text(result: Any, *, fallback: str) -> str:
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        return fallback

    @staticmethod
    def _chunk_text(text: str, limit: int) -> List[str]:
        if len(text) <= limit:
            return [text]
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(length, start + limit)
            split = end
            if end < length:
                for sep in ("\n\n", "\n", " "):
                    idx = text.rfind(sep, start, end)
                    if idx != -1 and idx > start:
                        split = idx + len(sep)
                        break
            if split <= start:
                split = end
            chunks.append(text[start:split])
            start = split
        return chunks

    def _log_warning(self, message: str) -> None:
        if tqdm and self.progress:
            tqdm.write(message)
        else:
            print(message)

    @staticmethod
    def save(df, out_path:Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".parquet":
            engine_kwargs = {}
            engine_label = PARQUET_ENGINE or "auto"
            if PARQUET_ENGINE:
                engine_kwargs["engine"] = PARQUET_ENGINE
            try:
                df.to_parquet(out_path, index=False, **engine_kwargs)
                print(f"✅ Parquet 저장({engine_label}): {out_path}")
                return
            except Exception as e:
                csv_path = out_path.with_suffix(".csv")
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(
                    f"⚠️ Parquet 엔진 실패({engine_label}) → CSV로 저장: {csv_path}\n"
                    f"   상세: {e}"
                )
                return
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ CSV 저장: {out_path}")


def _load_existing_corpus(path: Path) -> Optional["pd.DataFrame"]:
    if pd is None:
        return None
    candidates = [path]
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        candidates.append(path.with_suffix(".csv"))
    elif suffix == ".csv":
        candidates.append(path.with_suffix(".parquet"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".parquet":
                engine_kwargs = {}
                if PARQUET_ENGINE:
                    engine_kwargs["engine"] = PARQUET_ENGINE
                return pd.read_parquet(candidate, **engine_kwargs)
            return pd.read_csv(candidate)
        except Exception as exc:
            engine_label = PARQUET_ENGINE or "auto"
            print(
                f"⚠️ 기존 코퍼스 로드 실패 ({candidate}, engine={engine_label}): {exc}",
                flush=True,
            )
    return None


def _is_cache_fresh(cached: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if not cached.get("ok"):
        return False
    if not cached.get("text"):
        return False
    try:
        cached_size = int(cached.get("size", -1))
        row_size = int(row.get("size", -1))
    except (TypeError, ValueError):
        return False
    if cached_size != row_size:
        return False
    try:
        cached_mtime = float(cached.get("mtime", 0.0))
        row_mtime = float(row.get("mtime", 0.0))
    except (TypeError, ValueError):
        return False
    if abs(cached_mtime - row_mtime) > 1.0:
        return False
    return True


def _split_cache(
    file_rows: List[Dict[str, Any]],
    existing_df: Optional["pd.DataFrame"],
    *,
    force_paths: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional["pd.DataFrame"]]:
    if pd is None or existing_df is None or existing_df.empty or "path" not in existing_df.columns:
        return list(file_rows), None

    meta_map: Dict[str, Dict[str, Any]] = {}
    seen_paths: Set[str] = set()
    for rec in existing_df[["path", "size", "mtime"]].drop_duplicates(subset=["path"]).to_dict(orient="records"):
        key = str(rec.get("path") or "")
        if key:
            meta_map[key] = rec
            seen_paths.add(key)

    to_process: List[Dict[str, Any]] = []
    process_paths: Set[str] = set()
    for row in file_rows:
        path = str(row.get("path") or "")
        force = force_paths is not None and path in force_paths
        cached = meta_map.get(path)
        if force or not cached or not _is_cache_fresh(cached, row):
            to_process.append(row)
            if path:
                process_paths.add(path)

    if not process_paths:
        return to_process, existing_df.copy()

    mask = ~existing_df["path"].astype(str).isin(process_paths)
    remainder = existing_df[mask].copy()
    return to_process, remainder


def _collect_existing_rows(
    existing_df: Optional["pd.DataFrame"],
    target_paths: Set[str],
) -> Optional["pd.DataFrame"]:
    if pd is None or existing_df is None or existing_df.empty or not target_paths:
        return None
    mask = existing_df["path"].astype(str).isin({str(p) for p in target_paths})
    subset = existing_df[mask].copy()
    return subset if not subset.empty else None


# =========================
# 토픽 모델
# =========================
@dataclass
class TrainConfig:
    max_features: int = 50_000
    n_components: int = DEFAULT_N_COMPONENTS
    n_clusters: int = 30
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.8
    use_sentence_transformer: bool = True
    embedding_model: str = DEFAULT_EMBED_MODEL
    embedding_batch_size: int = 32
    async_embeddings: bool = True
    embedding_concurrency: int = 1
    embedding_dtype: str = "auto"
    # 대규모 코퍼스 처리용 청크/서브프로세스 임베딩 옵션
    embedding_chunk_size: int = 0  # 0이면 전체 한 번에
    embedding_chunk_start: int = 0  # chunk_size>0일 때 시작 청크 인덱스(포함)
    embedding_chunk_end: int = -1  # chunk_size>0일 때 끝 청크 인덱스(미포함, -1이면 끝까지)
    embedding_subprocess_fallback: bool = True


def _resolve_embed_dtype(cfg: TrainConfig) -> str:
    env_raw = os.getenv(EMBED_DTYPE_ENV)
    env_value = _sanitize_embed_dtype(env_raw)
    if env_raw:
        if env_value is not None:
            print(f"⚙️ 임베딩 dtype 설정: {env_value} ({EMBED_DTYPE_ENV})", flush=True)
            return env_value
        print(f"⚠️ {EMBED_DTYPE_ENV}={env_raw!r} 값이 잘못되어 auto 모드로 유지합니다.", flush=True)
    cfg_value = _sanitize_embed_dtype(getattr(cfg, "embedding_dtype", None))
    return cfg_value or "auto"

class TopicModel:
    def __init__(self, cfg:TrainConfig):
        if any(x is None for x in (TfidfVectorizer, TruncatedSVD, MiniBatchKMeans, Pipeline)):
            raise RuntimeError("scikit-learn 필요. pip install scikit-learn joblib")
        self.cfg=cfg
        self.pipeline:Optional[Pipeline]=None
        self._kmeans_n_init = _resolve_kmeans_n_init()

    def fit(self, df, text_col="text"):
        texts=(df[text_col].fillna("").astype(str)).tolist()
        print("🧠 학습 준비: TF-IDF → SVD → KMeans", flush=True)
        spin=Spinner(prefix="  학습 중")
        spin.start()
        try:
            self.pipeline = Pipeline(steps=[
                ("tfidf", TfidfVectorizer(
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=self.cfg.ngram_range,
                    max_features=self.cfg.max_features,
                    min_df=self.cfg.min_df,
                    max_df=self.cfg.max_df,
                )),
                ("svd", TruncatedSVD(n_components=self.cfg.n_components, random_state=42)),
                ("kmeans", MiniBatchKMeans(n_clusters=self.cfg.n_clusters, random_state=42, batch_size=2048, n_init=self._kmeans_n_init)),
            ])
            t0=time.time()
            self.pipeline.fit(texts)
            t1=time.time()
        finally:
            spin.stop()
        print(f"✅ 학습 완료 (docs={len(texts):,}, {t1-t0:.1f}s)", flush=True)
        return self

    def predict(self, df, text_col="text")->List[int]:
        texts=(df[text_col].fillna("").astype(str)).tolist()
        return self.pipeline.predict(texts)

    def transform(self, df, text_col="text"):
        texts=(df[text_col].fillna("").astype(str)).tolist()
        X=self.pipeline.named_steps["tfidf"].transform(texts)
        Z=self.pipeline.named_steps["svd"].transform(X)
        return Z

    def save(self, path:Path):
        if joblib is None: raise RuntimeError("joblib 필요")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cfg":self.cfg,"pipeline":self.pipeline}, path)


class SentenceBertModel:
    def __init__(self, cfg: TrainConfig):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers 라이브러리가 필요합니다. pip install sentence-transformers"
            )
        self.cfg = cfg
        self.model_name = cfg.embedding_model or DEFAULT_EMBED_MODEL
        print(f"🧠 Sentence-BERT 준비: {self.model_name}", flush=True)
        try:
            self._encoder = SentenceTransformer(self.model_name)
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            meta_issue = "meta tensor" in message or "to_empty" in message
            if meta_issue:
                print("⚠️ SentenceTransformer 로드 실패 → CPU 강제 시도", flush=True)
                try:
                    self._encoder = SentenceTransformer(self.model_name, device="cpu")
                except Exception as inner_exc:
                    raise RuntimeError(
                        "SentenceTransformer 초기화에 실패했습니다.\n"
                        "PyTorch를 README 권장 버전(torch 2.3.0, torchvision 0.18.0, torchaudio 2.3.0)으로 재설치해 주세요."
                    ) from inner_exc
            else:
                raise
        self.embedding_dim = int(self._encoder.get_sentence_embedding_dimension())
        self._target_dtype = (cfg.embedding_dtype or "auto").strip().lower()
        self._np_dtype = np.float16 if self._should_use_fp16() else np.float32
        self.cluster_model: Optional[MiniBatchKMeans] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self._kmeans_n_init = _resolve_kmeans_n_init()
        self._async_enabled = bool(getattr(self.cfg, "async_embeddings", False))
        self._async_threshold = max(512, int(self.cfg.embedding_batch_size) * 4)
        self._async_embedder = (
            AsyncSentenceEmbedder(
                self._encoder,
                batch_size=max(1, int(self.cfg.embedding_batch_size)),
                concurrency=max(1, int(getattr(self.cfg, "embedding_concurrency", 1))),
                target_dtype=self._target_dtype,
                device=self._encoder_device(),
            )
            if self._async_enabled
            else None
        )

    def _encoder_device(self) -> Optional[str]:
        device = getattr(self._encoder, "device", None)
        if device is None:
            device = getattr(self._encoder, "_target_device", None)
        if device is None:
            return None
        return str(device)

    def _should_use_fp16(self) -> bool:
        if self._target_dtype == "fp16":
            return True
        if self._target_dtype == "fp32":
            return False
        device = self._encoder_device()
        return bool(device and device.startswith("cuda"))

    def encode(self, texts: List[str], *, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        use_async = (
            self._async_enabled
            and self._async_embedder is not None
            and len(texts) >= self._async_threshold
        )
        if use_async:
            try:
                return self._async_embedder.encode(texts)
            except Exception as exc:
                print(f"⚠️ Async 임베딩 실패 → 동기 모드로 재시도합니다: {exc}", flush=True)
        embeddings = self._encoder.encode(
            texts,
            batch_size=max(1, int(self.cfg.embedding_batch_size)),
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.dtype != self._np_dtype:
            embeddings = embeddings.astype(self._np_dtype, copy=False)
        return embeddings

    def fit(self, df, text_col: str = "text") -> np.ndarray:
        texts = (df[text_col].fillna("").astype(str)).tolist()
        show_progress = tqdm is not None and len(texts) > 1000
        embeddings = self.encode(texts, show_progress=show_progress)

        can_cluster = (
            MiniBatchKMeans is not None
            and self.cfg.n_clusters > 0
            and embeddings.shape[0] >= max(10, self.cfg.n_clusters)
        )
        if can_cluster:
            print("🔖 클러스터링: MiniBatchKMeans", flush=True)
            self.cluster_model = MiniBatchKMeans(
                n_clusters=self.cfg.n_clusters,
                random_state=42,
                batch_size=2048,
                n_init=self._kmeans_n_init,
            )
            self.cluster_model.fit(embeddings)
            try:
                labels = self.cluster_model.labels_
            except AttributeError:
                labels = self.cluster_model.predict(embeddings)
            self.cluster_labels_ = np.asarray(labels, dtype=np.int32)
        else:
            self.cluster_model = None
            self.cluster_labels_ = None
            if MiniBatchKMeans is None:
                print("⚠️ scikit-learn MiniBatchKMeans 미설치로 토픽 라벨링을 건너뜁니다.", flush=True)
            elif embeddings.shape[0] < max(10, self.cfg.n_clusters):
                print("ℹ️ 문서 수가 적어 토픽 클러스터링을 건너뜁니다.", flush=True)
        return embeddings

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.cluster_model is None:
            raise RuntimeError("클러스터링 모델이 초기화되지 않았습니다.")
        labels = self.cluster_model.predict(embeddings)
        return np.asarray(labels, dtype=np.int32)

    def save(self, path: Path) -> None:
        if joblib is None:
            raise RuntimeError("joblib 필요. pip install joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "version": 2,
            "model_type": MODEL_TYPE_SENTENCE_TRANSFORMER,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "train_config": self.cfg,
        }
        if self.cluster_model is not None:
            payload["cluster_model"] = self.cluster_model
        joblib.dump(payload, path)


# =========================
# 청크 임베딩 (서브프로세스 GPU→CPU fallback)
# =========================
def _run_embed_chunk_subprocess(
    texts: List[str],
    cfg: TrainConfig,
    chunk_id: int,
    total_chunks: int,
) -> np.ndarray:
    """Embed a chunk of texts in a subprocess to isolate MPS OOM; CPU로 재시도."""
    root = Path(__file__).resolve().parents[2]
    cli_path = root / "scripts" / "pipeline" / "infopilot.py"
    if not cli_path.exists():
        raise RuntimeError(f"embed-chunk 명령을 찾을 수 없습니다: {cli_path}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_path = td_path / "chunk.json"
        output_path = td_path / "embeddings.npy"
        input_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")

        base_cmd = [
            sys.executable,
            str(cli_path),
            "embed-chunk",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            cfg.embedding_model,
            "--batch-size",
            str(max(1, int(cfg.embedding_batch_size))),
            "--concurrency",
            str(max(1, int(cfg.embedding_concurrency))),
            "--dtype",
            cfg.embedding_dtype or "auto",
        ]
        # 청크 임베딩은 안정성을 위해 기본적으로 동기 모드로 강제한다.
        base_cmd.append("--no-async")

        def _run(cmd, env=None) -> int:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            if proc.returncode != 0:
                stdout = (proc.stdout or b"").decode(errors="ignore")
                stderr = (proc.stderr or b"").decode(errors="ignore")
                print(
                    f"⚠️ 청크 임베딩 실패(chunk {chunk_id}/{total_chunks-1}, rc={proc.returncode})"
                    f"\nstdout: {stdout[:2000]}\nstderr: {stderr[:2000]}",
                    flush=True,
                )
            return proc.returncode

        rc = _run(base_cmd)
        if rc != 0 and cfg.embedding_subprocess_fallback:
            env = os.environ.copy()
            env["INFOPILOT_FORCE_CPU"] = "1"
            print(f"⚠️ chunk {chunk_id} → CPU로 재시도합니다.", flush=True)
            rc = _run(base_cmd, env=env)

        if rc != 0:
            raise RuntimeError(f"청크 임베딩 실패(chunk {chunk_id})")

        emb = np.load(output_path)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32, copy=False)
        return emb


def _chunked_sentence_embeddings(texts: List[str], cfg: TrainConfig) -> np.ndarray:
    chunk_size = max(1, int(cfg.embedding_chunk_size))
    start_chunk = max(0, int(getattr(cfg, "embedding_chunk_start", 0) or 0))
    end_chunk = int(getattr(cfg, "embedding_chunk_end", -1) or -1)
    total_chunks = math.ceil(len(texts) / chunk_size) if texts else 0
    embeddings_list: List[np.ndarray] = []
    progress = ProgressLine(total=max(1, total_chunks), label="Chunk 임베딩", update_every=1)

    for chunk_id, start_idx in enumerate(range(0, len(texts), chunk_size)):
        if chunk_id < start_chunk:
            continue
        if end_chunk >= 0 and chunk_id >= end_chunk:
            break
        chunk_texts = texts[start_idx : start_idx + chunk_size]
        emb = _run_embed_chunk_subprocess(chunk_texts, cfg, chunk_id, total_chunks)
        embeddings_list.append(emb)
        progress.update()

    progress.close()
    if not embeddings_list:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(embeddings_list, axis=0)


def _fit_sentence_transformer_chunked(train_df, text_col: str, cfg: TrainConfig):
    """Chunk + subprocess 기반 임베딩 후 클러스터/메트릭 계산."""
    semantic_model = SentenceBertModel(cfg)
    texts = (train_df[text_col].fillna("").astype(str)).tolist()
    embeddings = _chunked_sentence_embeddings(texts, cfg)

    metrics: Dict[str, float] = {}
    labels: Optional[np.ndarray] = None

    can_cluster = (
        MiniBatchKMeans is not None
        and cfg.n_clusters > 0
        and embeddings.shape[0] >= max(10, max(1, cfg.n_clusters))
    )
    if can_cluster:
        print("🔖 클러스터링: MiniBatchKMeans", flush=True)
        cluster_model = MiniBatchKMeans(
            n_clusters=cfg.n_clusters,
            random_state=42,
            batch_size=2048,
            n_init=_resolve_kmeans_n_init(),
        )
        cluster_model.fit(embeddings)
        try:
            labels = cluster_model.labels_
        except AttributeError:
            labels = cluster_model.predict(embeddings)
        labels = np.asarray(labels, dtype=np.int32)
        semantic_model.cluster_model = cluster_model
        semantic_model.cluster_labels_ = labels
        metrics = evaluate_embeddings(embeddings, labels, topk=min(5, max(1, embeddings.shape[0] - 1)))
    else:
        semantic_model.cluster_model = None
        semantic_model.cluster_labels_ = None
        if MiniBatchKMeans is None:
            print("⚠️ scikit-learn MiniBatchKMeans 미설치로 토픽 라벨링을 건너뜁니다.", flush=True)
        elif embeddings.shape[0] < max(10, max(1, cfg.n_clusters)):
            print("ℹ️ 문서 수가 적어 토픽 클러스터링을 건너뜁니다.", flush=True)

    return embeddings, semantic_model, metrics


# =========================
# 파이프라인 실행 (메인 함수)
# =========================

def run_step2(
    file_rows: List[Dict[str, Any]],
    out_corpus: Path = Path("./corpus.parquet"),
    out_model: Path = Path("./topic_model.joblib"),
    cfg: TrainConfig = TrainConfig(),
    use_tqdm: bool = True,
    translate: bool = False,
    *,
    scan_state_path: Optional[Path] = None,
    chunk_cache_path: Optional[Path] = None,
    skip_extract: bool = False,
    train_embeddings: bool = True,
):
    global tqdm
    original_tqdm = tqdm
    if not use_tqdm:
        tqdm = None

    if cfg is None:
        cfg = TrainConfig()
    else:
        cfg = replace(cfg)

    target_embed_dtype = _resolve_embed_dtype(cfg)
    cfg.embedding_dtype = target_embed_dtype

    chunk_cache = _create_chunk_cache(chunk_cache_path) if chunk_cache_path else None
    scan_state = load_scan_state(scan_state_path) if scan_state_path else None

    try:
        print("=== Step 2 시작: 내용 추출 & 학습 === (번역: " + ("활성" if translate else "비활성") + ")", flush=True)
        t_all = time.time()
        if pd is None:
            raise RuntimeError("pandas 필요")

        total_count = len(file_rows)
        cached_by_state = 0
        force_paths: Optional[Set[str]] = None

        existing_df = _load_existing_corpus(out_corpus)

        if skip_extract:
            if existing_df is None or existing_df.empty:
                raise RuntimeError(
                    "skip_extract가 설정되었지만 기존 corpus가 없습니다. 먼저 추출을 포함한 pipeline/train을 실행해 corpus.parquet을 생성하세요."
                )
            print("⏭️ 추출 스킵: 기존 corpus를 그대로 사용합니다.", flush=True)
            df = existing_df.copy()
            _prepare_text_frame(df)
            to_process = []
            reused_df = None
            df_new = df.copy()
            df_new_chunks = df.copy()
            process_count = 0
        else:
            if scan_state_path and scan_state is not None:
                forced_rows, cached_rows = filter_incremental_rows(file_rows, scan_state)
                force_paths = {str(row.get("path") or "") for row in forced_rows if row.get("path")}
                cached_by_state = len(cached_rows)
                if force_paths:
                    print(
                        f"⚙️ 증분 상태: {len(force_paths):,}건 재처리, 캐시 일치 {cached_by_state:,}건",
                        flush=True,
                    )
                else:
                    print("⚙️ 증분 상태: 신규 변경 없음", flush=True)

            to_process, reused_df = _split_cache(file_rows, existing_df, force_paths=force_paths)
            process_paths = {str(row.get("path") or "") for row in to_process if row.get("path")}
            process_count = len(process_paths)
            print(
                f"🗃️ 신규/변경 추출 대상: {process_count:,} | 총 스캔: {total_count:,}",
                flush=True,
            )

            if process_count == 0:
                if reused_df is not None:
                    df = reused_df.copy()
                elif existing_df is not None:
                    df = existing_df.copy()
                else:
                    df = pd.DataFrame(columns=list(ExtractRecord.__annotations__.keys()))
                _prepare_text_frame(df)
                order_map = {row["path"]: idx for idx, row in enumerate(file_rows)} if file_rows else {}
                if "path" in df.columns and order_map:
                    df["_order"] = df["path"].map(order_map)
                    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
                df = _deduplicate_corpus(df)
                CorpusBuilder.save(df, out_corpus)
                if chunk_cache:
                    chunk_cache.update_from_frame(df)
                    chunk_cache.save()
                if scan_state_path:
                    updated_state = update_scan_state(scan_state or {}, file_rows)
                    save_scan_state(scan_state_path, updated_state)
                df.attrs["metrics"] = {}
                df.attrs["incremental"] = {
                    "requested": process_count,
                    "effective": 0,
                    "skipped_by_state": cached_by_state,
                    "total": total_count,
                }
                df.attrs["target_embed_dtype"] = target_embed_dtype
                print("✨ 변경된 문서가 없어 기존 모델을 유지합니다.", flush=True)
                return df, None

            cb = CorpusBuilder(
                max_text_chars=200_000,
                progress=use_tqdm,
                translate=translate,
                target_embed_dtype=target_embed_dtype,
                # PyMuPDF가 다중 스레드에서 불안정하므로 macOS 기본은 워커 1개로 제한
                max_workers=int(os.getenv("INFOPILOT_MAX_EXTRACT_WORKERS", "1")),
            )
            df_new = cb.build(to_process) if process_count else pd.DataFrame(columns=list(ExtractRecord.__annotations__.keys()))

        restored_df = None
        unchanged_paths: Set[str] = set()
        if chunk_cache and df_new is not None and not df_new.empty:
            unchanged_paths = chunk_cache.unchanged_paths(df_new)
            if unchanged_paths:
                print(f"♻️ 내용 해시 동일 문서 재사용: {len(unchanged_paths):,}", flush=True)
                if existing_df is not None:
                    restored_df = _collect_existing_rows(existing_df, unchanged_paths)
                df_new = df_new[~df_new["path"].isin(list(unchanged_paths))]

            df_new_chunks = (
                _apply_uniform_chunks(
                    df_new,
                    min_tokens=DEFAULT_CHUNK_MIN_TOKENS,
                    max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
                )
                if df_new is not None and not df_new.empty
                else pd.DataFrame(columns=list(df_new.columns) if df_new is not None else list(ExtractRecord.__annotations__.keys()))
            )
            if hasattr(df_new_chunks, "attrs"):
                df_new_chunks.attrs["target_embed_dtype"] = target_embed_dtype

            frames: List["pd.DataFrame"] = []
            if reused_df is not None and not reused_df.empty:
                frames.append(reused_df)
            if restored_df is not None and not restored_df.empty:
                frames.append(restored_df)
            if df_new_chunks is not None and not df_new_chunks.empty:
                frames.append(df_new_chunks)

            if frames:
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.DataFrame(columns=list(ExtractRecord.__annotations__.keys()))

            _prepare_text_frame(df)

            order_map = {row["path"]: idx for idx, row in enumerate(file_rows)} if file_rows else {}
            if "path" in df.columns and order_map:
                df["_order"] = df["path"].map(order_map)
                df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

            if "ok" in df.columns:
                df["ok"] = df["ok"].apply(lambda v: bool(v) if isinstance(v, bool) else str(v).strip().lower() in {"true", "1", "yes"})
            if "topic" in df.columns:
                df = df.drop(columns=["topic"])

        text_col = MODEL_TEXT_COLUMN if MODEL_TEXT_COLUMN in df.columns else "text"
        text_mask = df[text_col].fillna("").str.len() > 0
        train_df = df[df["ok"] & text_mask].copy()
        if not train_df.empty:
            _prepare_text_frame(train_df)
        print(f"🧹 학습 대상 문서: {len(train_df):,}/{len(df):,}", flush=True)
        if len(train_df) == 0:
            df = _deduplicate_corpus(df)
            CorpusBuilder.save(df, out_corpus)
            if scan_state_path:
                updated_state = update_scan_state(scan_state or {}, file_rows)
                save_scan_state(scan_state_path, updated_state)
            if chunk_cache:
                chunk_cache.update_from_frame(df)
                chunk_cache.save()
            print(f"⚠️ 유효 텍스트 없음. 코퍼스만 저장: {out_corpus}", flush=True)
            df.attrs["metrics"] = {}
            df.attrs["incremental"] = {
                "requested": process_count,
                "effective": 0,
                "skipped_by_state": cached_by_state,
                "total": total_count,
            }
            df.attrs["target_embed_dtype"] = target_embed_dtype
            return df, None

        if not train_embeddings:
            df = _deduplicate_corpus(df)
            CorpusBuilder.save(df, out_corpus)
            if scan_state_path:
                updated_state = update_scan_state(scan_state or {}, file_rows)
                save_scan_state(scan_state_path, updated_state)
            if chunk_cache:
                chunk_cache.update_from_frame(df)
                chunk_cache.save()
            print(f"📦 추출만 완료 (임베딩/모델 건너뜀): {out_corpus}", flush=True)
            df.attrs["metrics"] = {}
            df.attrs["incremental"] = {
                "requested": process_count,
                "effective": 0,
                "skipped_by_state": cached_by_state,
                "total": total_count,
            }
            df.attrs["target_embed_dtype"] = target_embed_dtype
            return df, None

        topics_df = None
        model_obj: Optional[Any] = None
        metrics: Dict[str, float] = {}

        if cfg.use_sentence_transformer and SentenceTransformer is not None:
            try:
                if cfg.embedding_chunk_size and cfg.embedding_chunk_size > 0:
                    embeddings, semantic_model, metrics = _fit_sentence_transformer_chunked(
                        train_df, text_col, cfg
                    )
                else:
                    semantic_model = SentenceBertModel(cfg)
                    embeddings = semantic_model.fit(train_df, text_col=text_col)
                    if semantic_model.cluster_labels_ is not None:
                        metrics = evaluate_embeddings(
                            embeddings,
                            semantic_model.cluster_labels_,
                            topk=min(5, max(1, embeddings.shape[0] - 1)),
                        )
                print(
                    f"✅ Sentence-BERT 임베딩 완료 (docs={embeddings.shape[0]:,}, dim={semantic_model.embedding_dim})",
                    flush=True,
                )
                if semantic_model.cluster_labels_ is not None:
                    train_df["topic"] = semantic_model.cluster_labels_
                    topics_df = train_df[["path", "topic"]].copy()
                model_obj = semantic_model
            except Exception as exc:
                raise RuntimeError(
                    f"Sentence-BERT 임베딩에 실패했습니다. TF-IDF 백업을 사용하지 않습니다: {exc}"
                ) from exc
        elif cfg.use_sentence_transformer and SentenceTransformer is None:
            raise RuntimeError("sentence-transformers가 설치되어 있지 않아 임베딩을 진행할 수 없습니다.")

        if model_obj is None:
            tm = TopicModel(cfg)
            tm.fit(train_df, text_col=text_col)
            labels = tm.predict(train_df, text_col=text_col)
            train_df["topic"] = labels
            topics_df = train_df[["path", "topic"]].copy()
            model_obj = tm
            metrics = {}

        if topics_df is not None:
            df = df.merge(topics_df, on="path", how="left")

        df = _deduplicate_corpus(df)
        CorpusBuilder.save(df, out_corpus)

        if isinstance(model_obj, SentenceBertModel):
            model_obj.save(out_model)
        elif isinstance(model_obj, TopicModel) and joblib:
            model_obj.save(out_model)

        if chunk_cache:
            current_paths = set(df["path"].astype(str)) if "path" in df.columns else set()
            missing = chunk_cache.known_paths() - current_paths
            if missing:
                chunk_cache.drop_paths(missing)
            chunk_cache.update_from_frame(df)
            chunk_cache.save()

        if scan_state_path:
            updated_state = update_scan_state(scan_state or {}, file_rows)
            save_scan_state(scan_state_path, updated_state)

        dt_all = time.time() - t_all
        print(f"💾 저장 완료: corpus → {out_corpus} | model → {out_model}", flush=True)
        print(f"🎉 Step 2 종료 (총 {dt_all:.1f}s)", flush=True)
        df.attrs["metrics"] = metrics or {}
        df.attrs["incremental"] = {
            "requested": process_count,
            "effective": len(df_new_chunks["path"].unique()) if not df_new_chunks.empty else 0,
            "skipped_by_state": cached_by_state,
            "total": total_count,
        }
        df.attrs["target_embed_dtype"] = target_embed_dtype
        return df, model_obj
    finally:
        tqdm = original_tqdm



def update_corpus_file(
    new_records: "pd.DataFrame",
    corpus_path: Path,
) -> "pd.DataFrame":
    """Merge `new_records` into the persisted corpus and return the updated frame."""
    if pd is None:
        raise RuntimeError("pandas 필요. pip install pandas")

    existing = _load_existing_corpus(corpus_path)
    if existing is None or existing.empty:
        combined = new_records.copy()
    else:
        if "path" in existing.columns and "path" in new_records.columns:
            paths_to_replace = set(new_records["path"].astype(str).tolist())
            mask = ~existing["path"].astype(str).isin(paths_to_replace)
            combined = pd.concat([existing[mask], new_records], ignore_index=True)
        else:
            combined = pd.concat([existing, new_records], ignore_index=True)

    combined = _apply_uniform_chunks(
        combined,
        min_tokens=DEFAULT_CHUNK_MIN_TOKENS,
        max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
    )
    _prepare_text_frame(combined)
    combined = _deduplicate_corpus(combined)
    CorpusBuilder.save(combined, corpus_path)
    return combined


def remove_from_corpus(paths: List[str], corpus_path: Path) -> "pd.DataFrame":
    """Remove documents whose paths match `paths` from the persisted corpus."""
    if pd is None:
        raise RuntimeError("pandas 필요. pip install pandas")

    existing = _load_existing_corpus(corpus_path)
    if existing is None or existing.empty:
        return pd.DataFrame(columns=list(ExtractRecord.__annotations__.keys()))

    to_drop = {str(p) for p in paths}
    if "path" not in existing.columns:
        return existing

    filtered = existing[~existing["path"].astype(str).isin(to_drop)].copy()
    filtered = _apply_uniform_chunks(
        filtered,
        min_tokens=DEFAULT_CHUNK_MIN_TOKENS,
        max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
    )
    _prepare_text_frame(filtered)
    filtered = _deduplicate_corpus(filtered)
    CorpusBuilder.save(filtered, corpus_path)
    return filtered
