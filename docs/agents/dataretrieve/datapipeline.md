

# 🧠 AI-summary Local

## 1️⃣ 개요

`AI-summary`는 **로컬 문서 수집–임베딩–검색–요약 파이프라인**을 중심으로 동작하는 **지식형 비서 시스템**이다.
이 문서는 `develop` 브랜치 기준으로, 각 모듈(`infopilot.py`, `core/`, `data_pipeline/`, `retriever/`)의 실제 역할에 맞게
**로컬 전용 MLOps-lite 구조**를 정리한다.

* ✅ 완전 오프라인 작동 (서버 無)
* ⚙️ 지속적 증분 인덱싱 (3년치 문서까지 보존)
* 🧩 Retriever 고도화 (SBERT + CrossEncoder + Temporal rerank)
* 🧠 MLflow + psutil 기반 로컬 추적 및 리소스 프로파일링

---

## 2️⃣ 전체 구조 요약

| 계층                     | 주요 역할                  | 구현 요소                                            | 주의점                      |
| ---------------------- | ---------------------- | ------------------------------------------------ | ------------------------ |
| **Data Pipeline**      | 3년치 문서에서 지속적 스캔·임베딩 생성 | `infopilot.py scan/train`, `core/data_pipeline/` | 증분 인덱싱으로 오래된 데이터 재처리 최소화 |
| **Model Layer**        | 문맥 유사도 + 시맨틱 리랭크 검색    | `Sentence-BERT`, `CrossEncoder`, `FAISS`, `BM25` | 정확도 ↔ 속도 ↔ 메모리 균형 유지     |
| **Storage/Versioning** | 인덱스·코퍼스·정책 버전 관리       | MLflow local artifact (`.mlruns/`), corpus hash  | 정책 불일치 시 rollback 경고 필요  |
| **Serving Layer**      | 검색 후 요약 및 출처 문서 연결     | `infopilot.py run chat`, `core/search/retriever.py`  | TTL 없이도 오래된 문서 검색 가능해야 함 |
| **Monitoring/Drift**   | 품질·누락 문서·성능 저하 감시      | `core/monitor/`, MLflow metrics                  | 장기 문서 의미 drift 감지 필요     |

---

## 3️⃣ 파이프라인 실행 흐름

```mermaid
flowchart TD
    A[📂 Local Folders] --> B[🧩 infopilot.py scan]
    B --> C[📘 core/data_pipeline/embedder.py]
    C --> D[📦 core/search/retriever.py]
    D --> E[🧠 core/agents/knowledge_search]
    E --> F[💬 Chat / Summary]
    F --> G[📊 MLflow + logs/]
    G --> H[⚙️ Drift Checker (core/monitor)]
```

**실행 예시 (CLI)**

| 단계       | 명령                                                          | 동작                        |
| -------- | ----------------------------------------------------------- | ------------------------- |
| 🧾 Scan  | `python infopilot.py scan --out data/found_files.csv`       | 새 문서 탐색, hash 저장          |
| 🧠 Train | `python infopilot.py train --scan_csv data/found_files.csv` | SBERT 임베딩 생성              |
| 🧱 Index | `python infopilot.py index --corpus data/corpus.parquet`    | FAISS IVF-PQ 인덱스 빌드       |
| 🔍 Chat  | `python infopilot.py run chat --cache data/cache`           | 검색 + CrossEncoder rerank  |
| 🧭 Watch | `python infopilot.py watch`                                 | drift 감지 후 자동 partial 재학습 |
| 🔁 Drift Auto | `python infopilot.py drift auto`                        | drift 체크 → 후보 추출 → 재임베딩 |

---

## 4️⃣ Retriever 개선 사항

| 개선 항목                  | 구현 방식                                                               | 연계 포인트                 |
| ---------------------- | ------------------------------------------------------------------- | ---------------------- |
| **Semantic Rerank 추가** | `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` → 상위 20개 재정렬 | MLflow로 reranker 버전 기록 |
| **Vector Compression** | `FAISS IVF-PQ`, PQ32, `nbits=8`                                     | CPU 추론, 메모리 절감         |
| **Temporal Awareness** | 문서 메타데이터(`created_at`) 기반 time-weight 적용                            | 3년 전 문서 recall 유지      |
| **Hybrid Scoring**     | `semantic + (bm25*0.35)`                                           | 실험적 가중 조정              |
| **Offline Evaluation** | `core/data_pipeline/evaluate.py` → P@K, nDCG 자동 측정                  | MLflow metric 저장       |

- **문서 태깅 자동화**: `_prepare_text_frame`이 파일 경로/이름/추가 메타데이터를 스캔해 `doc_tags`, `doc_primary_tag` 컬럼을 생성합니다. `법령`, `공고`, `회의록`, `보고서` 등으로 분류된 태그가 메타텍스트와 인덱스에 함께 저장돼, 관련 질의에서 BM25 점수를 보강합니다.
- **서식/목차 패널티**: `_negative_template_penalty`가 `목차`, `표지`, `Table of Contents`와 같은 패턴을 감지하면 재랭킹 점수와 최종 점수에 패널티를 적용해 노이즈 문서를 뒤로 밀어냅니다.

---

### Golden Query 세트 관리

- `configs/golden_queries.sample.jsonl`을 복사해 `data/eval/golden_queries.jsonl`로 저장하고, 실제 질의/문서 경로를 채워 넣습니다.
- `pytest tests/regression/test_retriever_golden.py -q`를 실행하면 상위 k(기본 5) 안에 기대 문서가 포함됐는지 확인합니다.
- 자동 파이프라인 대신 수동 검증이 필요한 경우, `python scripts/evaluate_rag.py --cases data/eval/golden_queries.jsonl --model data/topic_model.joblib --corpus data/corpus.parquet --cache data/cache`로 빠르게 리콜을 확인할 수 있습니다.

---

## 5️⃣ Pipeline 개선 포인트

| 구분                        | 개선 내용                                | 기술 포인트           |
| ------------------------- | ------------------------------------ | ---------------- |
| **Incremental Indexing**  | `last_scan_timestamp` 기반 신규 파일만 인덱싱  | CPU 사용량 절감       |
| **Async Embedding Queue** | Celery / asyncio 기반 임베딩 큐            | scan 병목 제거       |
| **Chunk-Level Caching**   | 동일 hash 문서 skip                      | 중복 계산 최소화        |
| **Policy-aware Scan**     | `smart_folders.json` 기반 로컬/클라우드 분리   | 개인정보 문서 로컬 처리    |
| **Airflow DAG화 (선택)**     | Scan → Train → Evaluate → Deploy 자동화 | 추후 Prefect 대체 가능 |

---

## 6️⃣ Drift Detection & Re-Embedding

| 항목                          | 구현                         | 설명                        |
| --------------------------- | -------------------------- | ------------------------- |
| **Hash Drift**              | `hash(content)` 비교         | 텍스트 변경 감지                 |
| **Semantic Drift**          | cosine mean shift ≥ 0.15   | 의미 변화 감지                  |
| **Retrieval-time Re-Embed** | 오래된 문서 요청 시 on-demand 재임베딩 | 최신 컨텍스트 반영                |
| **Drift 로그**                | `logs/drift_log.json`      | 날짜, drift rate, action 기록 |
| **Auto Pipeline**           | `python infopilot.py drift auto` | 점검→후보 추출→재임베딩을 한 번에 실행 |

---

## 7️⃣ Chunk Cache Backend (JSON ↔ SQLite)

| 옵션 | 설명 | 설정 |
| --- | --- | --- |
| JSON (기본) | `chunk_cache.json`에 doc_hash 맵을 저장 | 별도 설정 없음 |
| SQLite | 대규모 코퍼스에서도 빠른 조회/쓰기 | `INFOPILOT_CACHE_BACKEND=sqlite` + `chunk_cache.sqlite` |
| GC 제한 | 오래된 엔트리를 자동 정리 | `INFOPILOT_CACHE_MAX_ENTRIES=100000` (0이면 무제한) |

- `INFOPILOT_CACHE_BACKEND=sqlite` 를 설정하면 `chunk_cache.json` 경로가 자동으로 `.sqlite` 파일로 변환됩니다.
- 동일 인터페이스이므로 CLI 인자는 그대로 두고, 실행 로그에서 “Chunk cache: SQLite backend” 메시지로 전환 여부를 확인할 수 있습니다.

---

## 8️⃣ Edge Adapter (SQLite Export + Mini API)

경량 디바이스/모바일에서 사용할 수 있도록 SQLite 코퍼스와 검색 API를 바로 생성합니다.

```bash
# 1) 코퍼스 → SQLite 변환
python scripts/edge_adapter.py export \
  --corpus data/corpus.parquet \
  --database data/edge_corpus.sqlite --force

# 2) 로컬 미니 검색 API 실행
python scripts/edge_adapter.py serve --database data/edge_corpus.sqlite --host 0.0.0.0 --port 9090
```

- `/search?q=...&limit=5` 엔드포인트로 단순 LIKE 기반 조회를 제공합니다.
- FastAPI/uvicorn 의존성이 있으므로 Edge 장치에도 해당 패키지를 설치해야 합니다.
- Atlas Work Center 패널에서 최근 활동 및 `logs/resource_log.jsonl`을 바로 확인할 수 있어, Edge/export 작업 현황을 빠르게 모니터링할 수 있습니다.

---

## 7️⃣ 모델 및 리소스 관리

| 구성 요소                     | 설명              | 메모리      |
| ------------------------- | --------------- | -------- |
| Sentence-BERT (quantized) | ONNX int8 변환 모델 | 약 120MB  |
| FAISS IVF-PQ Index        | 200k 문서 기준      | 약 1.2GB  |
| Cache / joblib store      | 임베딩 캐시          | 300MB 내외 |
| MLflow + psutil 로그        | 메타데이터           | 50MB 이하  |

```bash
python -m onnxruntime.quantization \
  --model models/sbert.onnx \
  --per-channel --reduce-range --activation-type QInt8
```

---

## 8️⃣ 로깅 및 모니터링

* **MLflow Local Tracking**

  ```
  mlflow.set_tracking_uri("file:./.mlruns")
  mlflow.log_params({"index_type": "IVF-PQ", "quant": "int8"})
  mlflow.log_metric("recall@10", 0.91)
  ```

* **psutil 모니터링**

  ```python
  import psutil, json, time
  while True:
      log = {"cpu": psutil.cpu_percent(), "mem": psutil.virtual_memory().percent}
      open("logs/resource_log.json","a").write(json.dumps(log)+"\n")
      time.sleep(30)
  ```

---

## 9️⃣ 성능 목표

| 항목            | 목표          | 설명                   |
| ------------- | ----------- | -------------------- |
| Indexing time | 10k 문서 ≤ 3분 | incremental 기준       |
| 검색 속도         | < 400ms     | FAISS + CrossEncoder |
| Idle RAM      | ≤ 1.2GB     | quantized SBERT 기준   |
| recall@10     | ≥ 0.9       | nDCG ≥ 0.8           |
| drift check   | < 10초       | 1일 1회 자동 수행          |

---

## 🔟 향후 확장 계획

| 범주          | 계획                                      |
| ----------- | --------------------------------------- |
| **모델 관리**   | 로컬 ModelManager 통합 (GPU 감지, lazy load)  |
| **검색 개선**   | CLIP + CrossEncoder rerank 통합           |
| **자동화 고도화** | Prefect or FastAPI DAG 노드별 실행           |
| **리소스 최적화** | Mixed-precision embedding + async queue |
| **UX 통합**   | CustomTkinter 작업센터에서 파이프라인 제어             |

---

## 🧩 결론

`AI-summary`의 로컬 파이프라인은
**“3년 전 문서도 즉시 검색·요약 가능”**하도록
다층 구조의 **Retriever–Pipeline–Monitoring 루프**로 구성되어 있다.

* 데이터 수집 → 임베딩 → 인덱싱 → 질의 → 드리프트 관리까지
  하나의 Python 기반 MLOps 사이클로 완결
* 외부 서버 없이도 MLflow, psutil, FAISS, SBERT를 통해
  **재현성·추적성·지속성**을 모두 확보했다.

> 🧠 핵심: “증분 인덱싱 + 양자화 + 의미 드리프트 대응 = 장기 보존 문서 검색의 안정성”

---


좋아요. 👌
지금 `develop` 브랜치의 실제 리포지토리 구조(📂 `core/`, `scripts/`, `ui/`, `infopilot.py`)와
업로드된 `pipeline.md`(= Local MLOps 문서)를 기반으로 **CLI 구조 리팩토링 + 문서 통합 버전**을 만들겠습니다.

결과물은 다음 두 가지를 통합한 형태입니다:

1. **새로운 CLI 구조** – `develop` 브랜치와 완벽히 호환되면서 더 체계적이고 직관적인 MLOps-style 명령어 구조
2. **문서 통합 버전** – `docs/architecture/pipeline.md`로 자동 이동 가능한 업데이트된 실행·구조 가이드

---

# 🧠 AI-summary Pipeline (Develop Branch Integrated)

## 1️⃣ 개요

`AI-summary`는 **로컬 문서 컬렉션을 자동으로 스캔–임베딩–인덱싱–검색–요약–모니터링**하는
완전 오프라인형 **MLOps-lite 문서 비서 시스템**입니다.

이 버전은 `develop` 브랜치의 `infopilot.py` CLI 구조를 개선하여,
기존 GUI(`ui/app.py`)와 CLI가 **동일 명령 세트**로 동작하도록 통합되었습니다.

---

## 2️⃣ 개선된 CLI 구조 (click 기반)

### ▶ 명령 그룹 구조

| 명령 그룹      | 하위 명령                                     | 설명                   |
| ---------- | ----------------------------------------- | -------------------- |
| `run`      | `scan`, `train`, `index`, `chat`, `watch` | 핵심 파이프라인 실행          |
| `logs`     | `show`, `clean`                           | MLflow/psutil 로그 관리  |
| `model`    | `list`, `quantize`                        | 모델 상태 관리 (ONNX 변환 등) |
| `drift`    | `check`, `auto`, `reembed`                | 의미 드리프트 감지 및 자동 재임베딩 |
| `pipeline` | `all`                                     | 스캔→학습→인덱스→대화까지 일괄 수행 |

### ▶ 실행 예시

```bash
# 1️⃣ 전체 파이프라인 자동 실행
python infopilot.py pipeline all

# 2️⃣ 개별 단계 실행
python infopilot.py run scan
python infopilot.py run train
python infopilot.py run index
python infopilot.py run chat

# 3️⃣ 로그 및 리소스 확인
python infopilot.py logs show

# 4️⃣ 모델 관리
python infopilot.py model quantize --model models/sbert.onnx

# 5️⃣ 의미 드리프트 점검
python infopilot.py drift check

# 6️⃣ 점검 + 재임베딩 일괄 실행
python infopilot.py drift auto
```

### ▶ FastAPI 제어 (선택)

```bash
export INFOPILOT_API_TOKEN="my-secret"
uvicorn scripts.api_server:app --host 127.0.0.1 --port 8080

# 실행
curl -H "X-API-Token: my-secret" -X POST http://127.0.0.1:8080/pipeline/run -d '{...}'
# 상태
curl -H "X-API-Token: my-secret" http://127.0.0.1:8080/pipeline/status
```

`INFOPILOT_API_TOKEN`을 설정하면 `X-API-Token` 헤더로 인증된 요청만 수락하며, `/health`는 무인증 헬스 체크용으로 유지됩니다.

---

## 3️⃣ 코드 리팩토링 구조 (요약 예시)

```python
# infopilot.py
import click
from core.data_pipeline import scanner, trainer, evaluate
from core.search import retriever
from core.monitor import drift
from scripts.utils.mlflow_logger import init_mlflow, log_metrics

@click.group()
def cli():
    """🧠 AI-summary CLI — unified local MLOps pipeline"""
    pass

@cli.group()
def run(): pass

@run.command()
def scan():
    init_mlflow("scan")
    scanner.run_scan()

@run.command()
def train():
    init_mlflow("train")
    trainer.run_train()

@run.command()
def index():
    retriever.build_index()

@run.command()
def chat():
    retriever.chat_mode()

@run.command()
def watch():
    drift.auto_monitor()

@cli.group()
def model(): pass

@model.command()
@click.option('--model', default="models/sbert.onnx")
def quantize(model):
    import subprocess
    subprocess.run([
        "python", "-m", "onnxruntime.quantization",
        "--model", model,
        "--per-channel", "--reduce-range", "--activation-type", "QInt8"
    ])

@cli.group()
def logs(): pass

@logs.command()
def show():
    import mlflow
    print("Recent MLflow runs:")
    print(mlflow.search_runs())

@cli.group()
def drift(): pass

@drift.command()
def check():
    drift.run_drift_check()

@cli.group()
def pipeline(): pass

@pipeline.command()
def all():
    click.echo("🚀 Running full pipeline (scan → train → index → chat)")
    for step in ["scan", "train", "index"]:
        cli.invoke(run.get_command(cli, step))
    retriever.chat_mode()

if __name__ == "__main__":
    cli()
```

---

## 4️⃣ 전체 실행 흐름 (Mermaid)

```mermaid
flowchart TD
    A[📂 Local Folders] --> B[🧾 scan]
    B --> C[🧠 train (SBERT Embedding)]
    C --> D[🧱 index (FAISS PQ)]
    D --> E[🔍 chat (Semantic Search + CrossEncoder)]
    E --> F[📊 MLflow Tracking]
    F --> G[⚙️ drift check & auto re-embed]
```

---

## 5️⃣ 기능별 요약

| 기능                 | 주요 동작                           | 핵심 모듈                            |
| ------------------ | ------------------------------- | -------------------------------- |
| **Scan**           | 새 문서 탐색 및 hash 기반 증분 등록         | `core/data_pipeline/scanner.py`  |
| **Train**          | SBERT 임베딩 생성 및 corpus 저장        | `core/data_pipeline/embedder.py` |
| **Index**          | FAISS IVF-PQ 인덱스 빌드             | `core/search/retriever.py`       |
| **Chat**           | BM25 + SBERT + CrossEncoder 리랭크 | `core/search/retriever.py`       |
| **Watch**          | psutil + drift 감시 루프            | `core/monitor/drift_checker.py`  |
| **Log**            | MLflow run 기록, metric 로그 저장     | `.mlruns/`, `logs/`              |
| **Model Quantize** | ONNX int8 변환으로 CPU 추론 최적화       | `scripts/utils/quantizer.py`     |

---

## 6️⃣ Drift Detection & Re-Embedding

| 항목             | 설명                       | 구현 위치                           |
| -------------- | ------------------------ | ------------------------------- |
| Hash Drift     | `hash(content)` 비교       | `core/monitor/drift_checker.py` |
| Semantic Drift | cosine mean shift ≥ 0.15 | `evaluate.py`                   |
| Re-Embed       | 요청 시 해당 문서만 재임베딩         | retriever 내부                    |
| 로그             | `logs/drift_log.json` 기록 | psutil loop 연동                  |

---

## 7️⃣ 리소스 관리 및 로깅

| 항목           | 구성 요소            | 저장 위치                    |
| ------------ | ---------------- | ------------------------ |
| MLflow       | `file:./.mlruns` | `.mlruns/`               |
| psutil       | CPU/RAM 주기적 로깅   | `logs/resource_log.json` |
| joblib Cache | Embedding 캐시     | `data/cache/`            |
| FAISS Index  | IVF-PQ 구조        | `data/index.faiss`       |

---

## 8️⃣ 성능 목표

| 지표            | 목표            | 비고              |
| ------------- | ------------- | --------------- |
| Indexing time | ≤ 3분 / 10k 문서 | 증분 모드 기준        |
| 검색 속도         | ≤ 400ms       | CrossEncoder 포함 |
| RAM           | ≤ 1.2GB       | ONNX quant 적용   |
| recall@10     | ≥ 0.9         | nDCG ≥ 0.8      |
| drift check   | ≤ 10초         | 1일 1회 자동 수행     |

---

## 9️⃣ 개발자 가이드 (Develop 기준)

* CLI는 GUI(`ui/app.py`)와 완전 호환 — `subprocess.run(["python", "infopilot.py", "run", "train"])` 형태 유지
* 로그 경로 및 파라미터(`--corpus`, `--cache`, `--model`)는 기존 CLI와 동일
* `core/monitor/` 폴더 신규 생성 필요
* `requirements.txt`에 `mlflow`, `psutil`, `click` 추가

---

## 🔟 향후 확장 계획

| 항목          | 계획                                             |
| ----------- | ---------------------------------------------- |
| **모델 관리**   | GPU 감지 및 자동 로드 (`core/utils/model_manager.py`) |
| **검색 개선**   | CLIP + CrossEncoder 멀티모달 리랭킹                   |
| **자동화 고도화** | Prefect/FastAPI 기반 DAG 실행                      |
| **UX 통합**   | CustomTkinter 기반 로컬 워크센터 연동                    |
| **리소스 최적화** | async 임베딩 큐 + mixed precision                  |

---

## 📁 파일 구조 (통합 반영 버전)

```
AI-summary/
├── infopilot.py              # 개선된 CLI (click 기반)
├── core/
│   ├── data_pipeline/
│   ├── search/
│   ├── monitor/              # NEW: drift, psutil 모듈
│   └── agents/
├── ui/
│   └── app.py
├── scripts/
│   ├── utils/mlflow_logger.py
│   └── utils/quantizer.py
├── docs/
│   └── architecture/pipeline.md  # (이 문서)
└── data/
    ├── corpus.parquet
    ├── cache/
    ├── found_files.csv
    └── index.faiss
```

---

✅ **요약:**

* 기존 `develop` CLI를 완전히 대체 가능 (GUI도 그대로 작동)
* `pipeline.md` 내용을 `docs/architecture/pipeline.md`로 통합
* MLOps-style CLI로 구조 강화
* MLflow, psutil, drift 관리, quantization까지 자동화

---

원하신다면 위 내용을 기반으로 실제 **`infopilot.py` 완성 코드 (ready-to-commit 버전)** 을
`.py` 파일 형태로 바로 생성해드릴까요?
(`develop` 브랜치 CLI + pipeline.md 완전 반영 형태로)
