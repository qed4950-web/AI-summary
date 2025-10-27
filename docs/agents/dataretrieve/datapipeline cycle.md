# 🧭 AI-summary Pipeline — Cycle별 리팩토링 진행 현황 (develop 기준)

> 2024-10-26 `develop` HEAD 의 실제 코드·모듈을 다시 확인해 본 결과를 토대로 문서를 갱신했습니다.  
> 각 Cycle은 설계 의도뿐 아니라 **현재 구현된 파일 경로**와 **남아 있는 TODO**를 함께 표기합니다.

---

## 🔹 Cycle 1 · Baseline 구조 통합
**상태:** ✅ 완료 (click CLI + GUI 연동 사용 중)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| CLI 구조 | `click` 기반 그룹/서브커맨드 (`scan/train/chat/pipeline/index/watch/schedule/...`) | `scripts/pipeline/infopilot.py` |
| 실행 명령 | `python infopilot.py run scan/train/...` → shim이 `scripts/pipeline/infopilot.py`를 로드 | `infopilot.py` (compat shim) |
| GUI 연동 | CustomTkinter 화면이 CLI 서브프로세스를 호출 | `ui/app.py`, `ui/screens/*` |
| 로그 구조 | MLflow + psutil 리소스 로거 초기화 (`_command_session`) | `scripts/utils/mlflow_logger.py`, `core/monitor/resource_logger.py` |

### 남은 작업
- 없음 — Atlas UI의 Work Center 패널에서 최근 활동과 리소스 로그가 표시되도록 처리되었습니다.

---

## 🔹 Cycle 2 · Data Pipeline 고도화
**상태:** ✅ 완료 (증분 추적 + Async 임베딩 배포)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Incremental Indexing | 스캔 상태(`data/scan_state.json`) + `core/data_pipeline/incremental.py`의 `filter_incremental_rows` 활용 | `core/data_pipeline/incremental.py`, `scripts/pipeline/infopilot.py` |
| Async Embedding Queue | `AsyncSentenceEmbedder`와 `SentenceBertModel` 연결, 대용량 시 자동 async | `core/data_pipeline/embedder.py`, `core/data_pipeline/pipeline.py` |
| Chunk Caching | doc hash 기반 JSON 캐시 (`chunk_cache.json`) 및 재사용 로직 | `core/data_pipeline/cache_manager.py` |
| Embedding Eval | P@K/nDCG 평가 헬퍼 + `run_step2` metrics | `core/data_pipeline/evaluate.py`, `core/data_pipeline/pipeline.py` |

### 남은 작업
- 없음 — pandas 미설치 시 CLI가 즉시 안내하고, `INFOPILOT_CACHE_MAX_ENTRIES`로 캐시 GC가 동작합니다.

---

## 🔹 Cycle 3 · Retriever / 검색 고도화
**상태:** ✅ 완료 (하이브리드 + rerank 가동)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Semantic Rerank | CrossEncoder(ms-marco-MiniLM-L-6-v2) 기반 rerank pipeline | `core/search/retriever.py` (`CrossEncoderReranker`) |
| Hybrid Scoring | Adaptive lexical weight + semantic similarity 결합 | `core/search/retriever.py` |
| Temporal Weight | 문서 메타데이터 기반 가중치 및 필터 | `core/search/retriever.py` |
| Index 관리 | 배경 스레드 로더/빌더 | `core/search/index_manager.py`, `core/search/retriever.py` |

### 남은 작업
- 없음 — GPU CrossEncoder 로드 실패 시 자동으로 CPU로 재시도합니다.

---

## 🔹 Cycle 4 · Drift & Monitoring 자동화
**상태:** ✅ 완료 (자동 점검 + 재임베딩 파이프라인)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Hash Drift | 스캔 CSV vs 코퍼스 diff, doc_hash 비교 | `core/monitor/drift_checker.py` |
| Semantic Drift | 임베딩 평균 벡터 추적 + baseline 저장 | `core/monitor/drift_checker.py` |
| Re-Embedding | `infopilot drift auto`가 점검→후보 선택→재임베딩까지 일괄 실행 | `scripts/pipeline/infopilot.py` |
| psutil Logging | `_command_session`이 `logs/resource_log.jsonl`에 주기적 기록 | `core/monitor/resource_logger.py` |
| MLflow Tracking | MLflow 세션 컨텍스트 도입 | `scripts/utils/mlflow_logger.py` |

### 남은 작업
- 없음 — Work Center 패널에서 `logs/resource_log.jsonl`을 즉시 조회할 수 있습니다.

---

## 🔹 Cycle 5 · MLOps & UX 통합
**상태:** ✅ 완료 (자동 오케스트레이션 + 배포 도구 정비)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Prefect Flow | `scan → train → index → evaluate` DAG 샘플 | `scripts/prefect_dag.py` |
| FastAPI Server | `/pipeline/*` + 토큰 기반 인증/헬스 체크 | `scripts/api_server.py` |
| Desktop 브릿지 | Atlas CustomTkinter 앱 + API 호출 | `ui/app.py`, `ui/api_client.py`, `scripts/api_server.py` |
| Model Manager | 모델 로더/캐시/참조 카운트 | `core/infra/models.py` |
| 배포 스크립트 | PyInstaller 래퍼 (`scripts/build_desktop_ui.*`) | README/문서에서 경로 안내 |

### 남은 작업
- 없음 — 빌드 스크립트(`build_desktop_ui.bat/.ps1`)가 `--sign-cmd`/`-SignCommand`로 코드 서명을 실행합니다.

---

## 🔹 Cycle 6 · Optimization & Future
**상태:** ✅ 완료 (경량 캐시 + Edge 어댑터)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Mixed Precision | `INFOPILOT_EMBED_DTYPE` + `AsyncSentenceEmbedder`가 FP16/FP32 자동 선택 | `core/data_pipeline/embedder.py`, `core/data_pipeline/pipeline.py` |
| ONNX Quantization | `infopilot model quantize`로 INT8/FP32 ONNX 생성 | `scripts/utils/quantizer.py`, `scripts/pipeline/infopilot.py` |
| Cache Optimization | `INFOPILOT_CACHE_BACKEND=sqlite` 설정 시 SQLite 캐시 백엔드 사용 | `core/data_pipeline/cache_manager.py`, `core/data_pipeline/pipeline.py` |
| Edge Adapter | `scripts/edge_adapter.py export/serve`로 SQLite 코퍼스 + 경량 검색 API 제공 | `scripts/edge_adapter.py` |

### 남은 작업
- 없음 — SQLite 캐시 + Edge Adapter가 포함되어 Optimized 파이프라인이 기본입니다.

---

## ✅ 종합 요약
| Cycle | 주제 | 상태 | 비고 |
| --- | --- | --- | --- |
| 1 | CLI 통합 | ✅ 완료 | click CLI + GUI 연결 |
| 2 | Data Pipeline | ✅ 완료 | 증분 + Async + evaluators |
| 3 | Retriever 고도화 | ✅ 완료 | CrossEncoder rerank, hybrid scoring |
| 4 | Drift/Monitoring | ✅ 완료 | `drift auto`로 점검→재임베딩 일괄 처리 |
| 5 | MLOps/UX | ✅ 완료 | Prefect + 토큰 보호 FastAPI + Atlas UI 정비 |
| 6 | Optimization | ✅ 완료 | Mixed precision + SQLite 캐시 + Edge adapter |

> 설계서에 없는 신규 기능이나 추가 Cycle이 생기면 동일한 형식으로 상태/파일/남은 작업을 꼭 갱신하세요.
