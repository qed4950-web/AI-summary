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
| GUI 연동 | Tkinter/Electron 화면이 CLI 서브프로세스를 호출 | `ui/app.py`, `ui/screens/*`, `ui/electron/*` |
| 로그 구조 | MLflow + psutil 리소스 로거 초기화 (`_command_session`) | `scripts/utils/mlflow_logger.py`, `core/monitor/resource_logger.py` |

### 남은 작업
- Work Center UX 문서는 있지만, `ui/screens/conversation_screen.py`에서 모든 엔트리가 노출되지는 않아 후속 정리가 필요.

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
- pandas 미설치 환경 대응(현재 optional)과 cache GC 주기 설정은 TODO로 남아 있음.

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
- GPU rerank 실패 시 graceful fallback 로깅은 되어 있지만 자동 재시도 전략은 추가 필요.

---

## 🔹 Cycle 4 · Drift & Monitoring 자동화
**상태:** 🟡 부분 완료 (수동 명령 제공, 완전 자동화는 TODO)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Hash Drift | 스캔 CSV vs 코퍼스 diff, doc_hash 비교 | `core/monitor/drift_checker.py` |
| Semantic Drift | 임베딩 평균 벡터 추적 + baseline 저장 | `core/monitor/drift_checker.py` |
| Re-Embedding | `infopilot drift reembed` 커맨드가 후보를 재임베딩 | `scripts/pipeline/infopilot.py` |
| psutil Logging | `_command_session`이 `logs/resource_log.jsonl`에 주기적 기록 | `core/monitor/resource_logger.py` |
| MLflow Tracking | MLflow 세션 컨텍스트 도입 | `scripts/utils/mlflow_logger.py` |

### 남은 작업
- 드리프트 감지 → 자동 재임베딩까지의 완전 파이프라인은 아직 수동 명령 체인.
- 리소스 로그 대시보드는 문서화만 되어 있고 UI 연동 미완.

---

## 🔹 Cycle 5 · MLOps & UX 통합
**상태:** 🟡 부분 완료 (도구는 존재, 운영 가이드는 보완 필요)

### 구현 현황
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Prefect Flow | `scan → train → index → evaluate` DAG 샘플 | `scripts/prefect_dag.py` |
| FastAPI Server | `/pipeline/*` REST 엔드포인트 제공 | `scripts/api_server.py` |
| Desktop 브릿지 | Electron + CustomTkinter 하이브리드 UI | `ui/electron/*`, `ui/app.py`, `ui/api_client.py` |
| Model Manager | 모델 로더/캐시/참조 카운트 | `core/infra/models.py` |

### 남은 작업
- Prefect/FAST API 배포 가이드, 헬스 체크, 인증 Hook 추가 필요.
- Electron 빌드 스크립트는 제공되지만 CI에 자동화되지 않았음.

---

## 🔹 Cycle 6 · Optimization & Future
**상태:** 🟠 진행 중 (일부 기능 반영, Edge/Cache roadmap 미완)

### 현재 구현
| 항목 | 실제 구현 내용 | 주요 파일 |
| --- | --- | --- |
| Mixed Precision | `INFOPILOT_EMBED_DTYPE` + `AsyncSentenceEmbedder`가 FP16/FP32 자동 선택 | `core/data_pipeline/embedder.py`, `core/data_pipeline/pipeline.py` |
| Async Embedding | 배치/동시성 튜닝 옵션 (`embedding_batch_size`, `embedding_concurrency`) | `core/data_pipeline/pipeline.py` |
| ONNX Quantization | `infopilot model quantize` → INT8/FP32 ONNX 내보내기 | `scripts/utils/quantizer.py`, `scripts/pipeline/infopilot.py` |
| Cache | Doc-hash 기반 JSON 캐시 (sqlite 전환은 아직) | `core/data_pipeline/cache_manager.py` |

### 남은 작업
- 문서에 언급됐던 `scripts/edge_adapter.py`, sqlite 하이브리드 캐시는 **아직 존재하지 않습니다.**
- 모바일/Edge 경량 REST는 `scripts/api_server.py` 기반으로만 제공 → 전용 어댑터, SQLite corpus 출력이 필요.
- 모델 압축/온디맨드 검색 KPI(“속도 1.8×” 등)는 아직 검증되지 않아 문서에서 제거했습니다.

---

## ✅ 종합 요약
| Cycle | 주제 | 상태 | 비고 |
| --- | --- | --- | --- |
| 1 | CLI 통합 | ✅ 완료 | click CLI + GUI 연결 |
| 2 | Data Pipeline | ✅ 완료 | 증분 + Async + evaluators |
| 3 | Retriever 고도화 | ✅ 완료 | CrossEncoder rerank, hybrid scoring |
| 4 | Drift/Monitoring | 🟡 부분 완료 | 수동 커맨드 제공, 자동화 예정 |
| 5 | MLOps/UX | 🟡 부분 완료 | Prefect/API/Electron 존재, 배포 가이드 필요 |
| 6 | Optimization | 🟠 진행 중 | Mixed precision/ONNX 완료, Edge & sqlite 캐시 미완 |

> 설계서에 없는 신규 기능이나 추가 Cycle이 생기면 동일한 형식으로 상태/파일/남은 작업을 꼭 갱신하세요.
