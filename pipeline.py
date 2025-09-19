# pipeline.py  (Step2: 추출 + 학습)
import os, re, sys, time, threading, platform, asyncio, csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Iterable

from encoding_utils import detect_file_encodings

# ---- 선택 의존성(있으면 사용) ----
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from googletrans import Translator
except Exception:
    Translator = None
try:
    import docx
except Exception:
    docx = None
try:
    import pptx
except Exception:
    pptx = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import textract
except Exception:
    textract = None
try:
    import importlib
    hwp5_dataio = importlib.import_module("hwp5.dataio")  # type: ignore
    hwp5_txt = importlib.import_module("hwp5.hwp5txt")  # type: ignore
except Exception:
    hwp5_dataio = None
    hwp5_txt = None
try:
    import joblib
except Exception:
    joblib = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.pipeline import Pipeline
except Exception:
    TfidfVectorizer = TruncatedSVD = MiniBatchKMeans = Pipeline = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


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
    def start(self):
        if self._t: return
        def _run():
            while not self._stop.wait(self.interval):
                frame = self.FRAMES[self._i % len(self.FRAMES)]
                self._i += 1
                sys.stdout.write(f"\r{self.prefix} {frame} ")
                sys.stdout.flush()
        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()
    def stop(self, clear=True):
        if not self._t: return
        self._stop.set()
        self._t.join()
        if clear:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

class ProgressLine:
    def __init__(self, total:int, label:str, update_every:int=10):
        self.total = max(1, total)
        self.label = label
        self.update_every = max(1, update_every)
        self.start = time.time()
        self.n = 0
    def update(self, k:int=1):
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
    def close(self):
        self.n = self.total
        self.update(0)
        sys.stdout.write("\n"); sys.stdout.flush()
    @staticmethod
    def _fmt(s: float)->str:
        if s==float("inf"): return "∞"
        m, sec = divmod(int(s), 60); h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# =========================
# 텍스트 클린
# =========================
class TextCleaner:
    _multi = re.compile(r"\s+")
    @classmethod
    def clean(cls, s:str)->str:
        if not s: return ""
        s = "".join(ch if ch.isprintable() or ch in "\t\n\r" else " " for ch in s)
        s = s.replace("\x00"," ")
        return cls._multi.sub(" ", s).strip()

TOKEN_PATTERN = r'(?u)(?:[가-힣]{1,}|[A-Za-z0-9]{2,})'


# =========================
# Extractors
# =========================
import traceback
from collections import defaultdict


class MissingDependencyError(RuntimeError):
    """Raised when an optional extractor dependency is unavailable."""


@dataclass
class ExtractionAttempt:
    engine: str
    ok: bool
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""


@dataclass
class ExtractionResult:
    ok: bool
    text: str
    meta: Dict[str, Any]
    attempts: List[ExtractionAttempt]


class ExtractorRegistry:
    def __init__(self):
        self._map: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)

    def register(self, extensions: Iterable[str], name: str, func):
        for ext in extensions:
            self._map[ext.lower()].append((name, func))

    def extract(self, path: Path) -> ExtractionResult:
        ext = path.suffix.lower()
        handlers = self._map.get(ext, [])
        attempts: List[ExtractionAttempt] = []
        if not handlers:
            attempts.append(ExtractionAttempt("unhandled", False, error="no extractor registered"))
            return ExtractionResult(False, "", {"error": "no extractor registered"}, attempts)

        for name, func in handlers:
            try:
                text, meta = func(path)
                if isinstance(text, bytes):
                    text = text.decode("utf-8", "ignore")
                if not isinstance(text, str):
                    text = str(text or "")
                text = TextCleaner.clean(text)
                meta = dict(meta or {})
                meta.setdefault("engine", name)
                if text:
                    attempt = ExtractionAttempt(name, True, text, meta)
                    attempts.append(attempt)
                    return ExtractionResult(True, text, meta, attempts)
                attempt = ExtractionAttempt(name, False, "", meta, error="empty text")
                attempts.append(attempt)
            except MissingDependencyError as e:
                attempts.append(ExtractionAttempt(name, False, "", {"engine": name}, error=str(e)))
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                attempts.append(ExtractionAttempt(name, False, "", {"engine": name}, error=str(e), traceback=tb))

        last_error = attempts[-1].error if attempts else "unknown"
        return ExtractionResult(False, "", {"error": last_error}, attempts)


EXTRACTOR_REGISTRY = ExtractorRegistry()


def _require_dependency(module_obj, friendly: str) -> None:
    if module_obj is None:
        raise MissingDependencyError(f"필요 모듈 '{friendly}'이(가) 설치되어 있지 않습니다.")


def _require_excel_engine(engine: str) -> None:
    import importlib

    candidates = {
        "openpyxl": ["openpyxl"],
        "xlrd": ["xlrd2", "xlrd"],
        "pyxlsb": ["pyxlsb"],
    }
    modules = candidates.get(engine, [engine])
    for mod in modules:
        try:
            importlib.import_module(mod)
            return
        except ModuleNotFoundError:
            continue
    friendly = "/".join(modules)
    raise MissingDependencyError(f"엑셀 엔진 '{engine}' 사용을 위한 모듈({friendly})이 설치되어 있지 않습니다.")


def _df_to_text(df) -> str:
    cols = " | ".join(map(str, df.columns.tolist()))
    rows: List[str] = []
    for _, row in df.head(50).iterrows():
        rows.append(" • " + " | ".join(map(lambda x: str(x), row.tolist())))
    return TextCleaner.clean(f"{cols}\n" + "\n".join(rows))


def _extract_csv(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(pd, "pandas")
    last_error: Optional[Exception] = None
    encoding_used: Optional[str] = None
    df = None
    for enc in detect_file_encodings(path):
        try:
            df = pd.read_csv(path, nrows=200, encoding=enc, engine="python", dtype=str, na_filter=False)
            encoding_used = enc
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break
    if df is None:
        raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "encoding detection failed")
    text = _df_to_text(df.fillna(""))
    meta: Dict[str, Any] = {
        "engine": "pandas",
        "columns": df.columns.tolist(),
        "rows_preview": min(200, len(df)),
    }
    if encoding_used:
        meta["encoding"] = encoding_used
    return text, meta


def _extract_excel(path: Path, engine: str) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(pd, "pandas")
    _require_excel_engine(engine)
    sheets = pd.read_excel(path, sheet_name=None, nrows=200, engine=engine, dtype=str)
    parts: List[str] = []
    for name, df_sheet in sheets.items():
        df_sheet = df_sheet.fillna("")
        parts.append(f"[Sheet:{name}]")
        parts.append(" | ".join(map(str, df_sheet.columns.tolist())))
        for _, row in df_sheet.head(50).iterrows():
            parts.append(" • " + " | ".join(map(lambda x: str(x), row.tolist())))
    text = TextCleaner.clean("\n".join(parts))
    return text, {"engine": "pandas", "sheets": list(sheets.keys()), "excel_engine": engine}


def _extract_text_file(path: Path) -> Tuple[str, Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for enc in detect_file_encodings(path):
        try:
            text = path.read_text(encoding=enc)
            return text, {"engine": "text", "encoding": enc}
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break
    raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "text encoding detection failed")


def _extract_pdf_plumber(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(pdfplumber, "pdfplumber")
    with pdfplumber.open(str(path)) as pdf:
        texts = [page.extract_text() or "" for page in pdf.pages]
    text = "\n".join(t for t in texts if t)
    return text, {"engine": "pdfplumber", "pages": len(texts)}


def _extract_pdf_pypdf2(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(PyPDF2, "PyPDF2")
    reader = PyPDF2.PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        try:
            chunk = page.extract_text() or ""
        except Exception:
            chunk = ""
        if chunk:
            texts.append(chunk)
    text = "\n".join(texts)
    return text, {"engine": "PyPDF2", "pages": len(reader.pages)}


def _extract_pdf_pdfminer(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(pdfminer_extract_text, "pdfminer.six")
    text = pdfminer_extract_text(str(path))
    return text, {"engine": "pdfminer"}


def _extract_textract(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(textract, "textract")
    data = textract.process(str(path))
    return data.decode("utf-8", "ignore"), {"engine": "textract"}


def _extract_docx(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(docx, "python-docx")
    document = docx.Document(str(path))
    text = "\n".join(par.text for par in document.paragraphs)
    return text, {"engine": "python-docx", "paras": len(document.paragraphs)}


def _extract_pptx(path: Path) -> Tuple[str, Dict[str, Any]]:
    _require_dependency(pptx, "python-pptx")
    presentation = pptx.Presentation(str(path))
    texts: List[str] = []
    for idx, slide in enumerate(presentation.slides, 1):
        parts: List[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                content = (shape.text or "").strip()
                if content:
                    parts.append(content)
        if parts:
            texts.append(f"[Slide {idx}] " + " ".join(parts))
    return "\n".join(texts), {"engine": "python-pptx", "slides": len(presentation.slides)}


def _extract_hwp(path: Path) -> Tuple[str, Dict[str, Any]]:
    if not (hwp5_dataio and hwp5_txt):
        raise MissingDependencyError("pyhwp 모듈이 필요합니다.")
    HWP5File = getattr(hwp5_dataio, "HWP5File", None)
    TxtExtractor = getattr(hwp5_txt, "HWP5Txt", None) or getattr(hwp5_txt, "HWP5TextExtractor", None)
    if not (HWP5File and TxtExtractor):
        raise MissingDependencyError("pyhwp 텍스트 추출기를 찾을 수 없습니다.")
    with path.open("rb") as fh:
        doc = HWP5File(fh)  # type: ignore
        extractor = TxtExtractor(doc)  # type: ignore
        if hasattr(extractor, "text") and callable(getattr(extractor, "text")):
            text = extractor.text()
        elif hasattr(extractor, "to_text") and callable(getattr(extractor, "to_text")):
            text = extractor.to_text()
        elif hasattr(extractor, "get_text") and callable(getattr(extractor, "get_text")):
            text = extractor.get_text()
        elif hasattr(extractor, "extract_text") and callable(getattr(extractor, "extract_text")):
            text = extractor.extract_text()
        else:
            text = ""
    if isinstance(text, (list, tuple)):
        text = "\n".join(map(str, text))
    return str(text), {"engine": "pyhwp"}


EXTRACTOR_REGISTRY.register([".csv"], "csv-pandas", _extract_csv)
EXTRACTOR_REGISTRY.register([".txt"], "text", _extract_text_file)
EXTRACTOR_REGISTRY.register([".xlsx", ".xlsm", ".xltx"], "excel-openpyxl", lambda p: _extract_excel(p, "openpyxl"))
EXTRACTOR_REGISTRY.register([".xls"], "excel-xlrd", lambda p: _extract_excel(p, "xlrd"))
EXTRACTOR_REGISTRY.register([".xlsb"], "excel-pyxlsb", lambda p: _extract_excel(p, "pyxlsb"))
EXTRACTOR_REGISTRY.register([".pdf"], "pdfplumber", _extract_pdf_plumber)
EXTRACTOR_REGISTRY.register([".pdf"], "PyPDF2", _extract_pdf_pypdf2)
EXTRACTOR_REGISTRY.register([".pdf"], "pdfminer", _extract_pdf_pdfminer)
EXTRACTOR_REGISTRY.register([".pdf"], "textract", _extract_textract)
EXTRACTOR_REGISTRY.register([".docx"], "python-docx", _extract_docx)
EXTRACTOR_REGISTRY.register([".docx", ".doc"], "textract", _extract_textract)
EXTRACTOR_REGISTRY.register([".pptx"], "python-pptx", _extract_pptx)
EXTRACTOR_REGISTRY.register([".pptx", ".ppt"], "textract", _extract_textract)
EXTRACTOR_REGISTRY.register([".hwp"], "pyhwp", _extract_hwp)
EXTRACTOR_REGISTRY.register([".hwp"], "textract", _extract_textract)


# =========================
# 코퍼스 빌더 (번역 기능 수정)
# =========================
@dataclass
class ExtractRecord:
    path:str
    ext:str
    ok:bool
    text:str
    meta:Dict[str,Any]
    size:Optional[int]=None
    mtime:Optional[float]=None
    content:str=""

class CorpusBuilder:
    def __init__(self, max_text_chars:int=200_000, progress:bool=True, translate:bool=False):
        self.max_text_chars=max_text_chars
        self.progress=progress
        self.translate = translate
        self.translator = Translator() if translate and Translator else None
        if translate and not self.translator:
            print("⚠️ 경고: 'googletrans' 라이브러리를 찾을 수 없어 번역 기능이 비활성화됩니다.")
            print("   해결: pip install googletrans==4.0.0-rc1")
        self.failures: List[Dict[str, Any]] = []

    def build(self, file_rows:List[Dict[str,Any]]):
        if pd is None: raise RuntimeError("pandas 필요. pip install pandas")
        total=len(file_rows)
        recs:List[ExtractRecord]=[]
        self.failures.clear()

        iterator = file_rows
        if self.progress and tqdm:
            desc = "📥 Extract & Translate" if self.translate else "📥 Extract"
            iterator = tqdm(file_rows, desc=desc, unit="file")
        else:
            print("📥 Extract 시작", flush=True)
            prog=ProgressLine(total, "extracting", update_every=max(1,total//100 or 1))

        for row in iterator:
            recs.append(self._extract_one(row))
            if not (self.progress and tqdm):
                prog.update(1)

        if self.progress and tqdm:
            iterator.close()
        else:
            prog.close()

        df = pd.DataFrame([r.__dict__ for r in recs])
        if not df.empty:
            df["text"] = df["text"].fillna("").astype(str)
            df["content"] = df["content"].fillna("").astype(str)
        ok = int(df["ok"].sum()) if len(df)>0 else 0
        fail = int((~df["ok"]).sum()) if len(df)>0 else 0
        print(f"✅ Extract 완료: ok={ok}, fail={fail}", flush=True)
        return df

    def _extract_one(self, row:Dict[str,Any])->ExtractRecord:
        p=Path(row["path"])
        ext=p.suffix.lower()
        result = EXTRACTOR_REGISTRY.extract(p)

        if not result.ok:
            attempt = result.attempts[-1] if result.attempts else None
            self.failures.append({
                "path": str(p),
                "ext": ext,
                "engine": attempt.engine if attempt else "",
                "error": attempt.error if attempt else "no extractor",
                "traceback": attempt.traceback if attempt else "",
            })
            return ExtractRecord(
                str(p),
                ext,
                False,
                "",
                result.meta,
                row.get("size"),
                row.get("mtime"),
                str(p),
            )

        original_text = (result.text or "")[: self.max_text_chars]
        preview = original_text[:500]
        if len(original_text) > 500:
            preview += "…"
        content = f"{str(p)}\n{preview}" if preview else str(p)

        text_for_model = original_text
        if self.translator and original_text.strip():
            try:
                translated = asyncio.run(self.translator.translate(original_text, dest='en'))
                text_for_model = translated.text or ""
            except Exception as e:
                msg = f"\n[경고] '{p.name}' 번역 실패. 원본 텍스트 사용. 오류: {e}"
                if tqdm and self.progress:
                    tqdm.write(msg)
                else:
                    print(msg)

        return ExtractRecord(
            str(p),
            ext,
            True,
            text_for_model,
            result.meta,
            row.get("size"),
            row.get("mtime"),
            content,
        )

    @staticmethod
    def save(df, out_path:Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".parquet":
            try:
                df.to_parquet(out_path, index=False)
                print(f"✅ Parquet 저장: {out_path}")
                return
            except Exception as e:
                csv_path = out_path.with_suffix(".csv")
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"⚠️ Parquet 엔진 없음 → CSV로 저장: {csv_path}\n   상세: {e}")
                return
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ CSV 저장: {out_path}")

    def export_failures(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "ext", "engine", "error", "traceback"])
            writer.writeheader()
            for row in self.failures:
                writer.writerow(row)


# =========================
# 토픽 모델
# =========================
@dataclass
class TrainConfig:
    max_features:int=200_000
    n_components:int=256
    n_clusters:int=30
    ngram_range:Tuple[int,int]=(1,2)
    min_df:int=2
    max_df:float=0.8

class TopicModel:
    def __init__(self, cfg:TrainConfig):
        if any(x is None for x in (TfidfVectorizer, TruncatedSVD, MiniBatchKMeans, Pipeline)):
            raise RuntimeError("scikit-learn 필요. pip install scikit-learn joblib")
        self.cfg=cfg
        self.pipeline:Optional[Pipeline]=None

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
                ("kmeans", MiniBatchKMeans(n_clusters=self.cfg.n_clusters, random_state=42, batch_size=2048, n_init="auto")),
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


# =========================
# 파이프라인 실행 (메인 함수)
# =========================
def run_step2(file_rows:List[Dict[str,Any]],
              out_corpus:Path=Path("./corpus.parquet"),
              out_model:Path=Path("./topic_model.joblib"),
              cfg:TrainConfig=TrainConfig(),
              use_tqdm:bool=True,
              translate:bool=False):
    global tqdm
    if not use_tqdm:
        tqdm=None

    print("=== Step 2 시작: 내용 추출 & 학습 === (번역: " + ("활성" if translate else "비활성") + ")", flush=True)
    t_all=time.time()

    # 1) Extract & Translate
    cb=CorpusBuilder(max_text_chars=200_000, progress=True, translate=translate)
    df=cb.build(file_rows)
    failures_path = out_corpus.parent / "extract_failures.csv"
    cb.export_failures(failures_path)
    print(f"🧾 추출 실패 로그: {failures_path} (rows={len(cb.failures)})", flush=True)

    # 2) 학습 데이터 필터
    if pd is None: raise RuntimeError("pandas 필요")
    train_df = df[df["ok"] & (df["text"].str.len()>0)].copy()
    print(f"🧹 학습 대상 문서: {len(train_df):,}/{len(df):,}", flush=True)
    if len(train_df)==0:
        cb.save(df, out_corpus)
        print(f"⚠️ 유효 텍스트 없음. 코퍼스만 저장: {out_corpus}", flush=True)
        return df, None

    # 3) Train
    tm=TopicModel(cfg)
    tm.fit(train_df, text_col="text")
    labels=tm.predict(train_df)
    train_df["topic"]=labels
    df = df.merge(train_df[["path","topic"]], on="path", how="left")

    # 4) 저장
    cb.save(df, out_corpus)
    if joblib:
        tm.save(out_model)

    dt_all=time.time()-t_all
    print(f"💾 저장 완료: corpus → {out_corpus} | model → {out_model}", flush=True)
    print(f"🎉 Step 2 종료 (총 {dt_all:.1f}s)", flush=True)
    return df, tm
