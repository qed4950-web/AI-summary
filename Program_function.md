# 프로그램 기능 정의서

## 1. 플랫폼 개요
- InfoPilot은 단일 CLI (`infopilot.py`)와 모듈형 에이전트로 구성된 로컬 우선 AI 비서입니다.
- 핵심 구성요소는 스마트 폴더 정책 엔진, 데이터 파이프라인(스캔·학습·증분), 검색 인덱스, 모델 캐시, 그리고 에이전트별 워크플로로 나뉩니다.
- 생성된 산출물은 기본적으로 `data/`(코퍼스·모델)와 `index_cache/`(검색 인덱스) 아래에 저장되며, 에이전트별 결과는 정책 또는 명령줄 인자에 따라 폴더가 분리됩니다.

## 2. 스마트 폴더 정책 엔진
- 정책 정의는 JSON 형식(`config/smart_folders.json`)으로 관리되며, 스키마는 `core/data_pipeline/policies/schema/smart_folder_policy.schema.json`에 존재합니다.
- `PolicyEngine`(`core/data_pipeline/policies/engine.py`)은 정책을 로드해 다음을 제공합니다.
  - 폴더별 허용 에이전트, 인덱싱 모드(`realtime`/`scheduled`/`manual`), 보안 처리 수준, 보존 정책.
  - `allows(Path, agent)`를 통해 스캔/학습/감시 시 문서 허용 여부를 판정.
  - `filter_records(...)`로 CSV 스캔 결과에서 정책에 부합하는 항목만 남깁니다.
- `ScheduleSpec.from_policy`는 `indexing` 섹션을 바탕으로 예약 주기를 파싱하며, `cron` 또는 `interval_minutes` 값을 해석합니다.

## 3. 데이터 파이프라인 및 인덱스
### 3.1 스캔 단계 (`infopilot.py scan`)
- `FileFinder`(`core/data_pipeline/filefinder.py`)가 허용 확장자(문서/코드/오피스/이미지 등)를 수집하고 메타데이터 CSV(`data/found_files.csv`)를 생성합니다.
- 정책이 지정되면 `PolicyEngine.filter_records`로 지식·검색 에이전트(`knowledge_search`) 허용 범위만 출력.

### 3.2 학습 단계 (`infopilot.py train`/`pipeline`)
- `run_step2`(`core/data_pipeline/pipeline.py`)가 문서를 텍스트로 추출하고 청크 단위로 정규화한 뒤 코퍼스(`data/corpus.parquet`)와 모델(`data/topic_model.joblib`)을 생성합니다.
- 기본 임베딩 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`DEFAULT_EMBED_MODEL`).
- 기본 설정은 토큰 200~500 범위로 문서를 분할하고, 다국어 Sentence-BERT 학습을 시도한 후 실패 시 TF-IDF+SVD 백업 경로를 사용합니다.
- `TrainConfig`는 최대 특성 수(50,000), SVD 차원(128), 클러스터 수(25) 등을 제공합니다.

### 3.3 인덱스 및 검색 캐시
- `VectorIndex`(`core/search/retriever.py`)는 문서 임베딩, BM25 토큰, 메타데이터, FAISS 인덱스를 관리합니다.
- `Retriever`는 임베딩 기반 k-NN 검색을 수행하고 선택적으로 Cross-Encoder 재랭킹을 적용합니다.
- Cross-Encoder 기본 모델: `cross-encoder/ms-marco-MiniLM-L-6-v2`; BM25 가중치는 `--lexical-weight`로 조절 가능합니다.
- `IndexManager`(`core/search/index_manager.py`)가 인덱스 로딩/재빌드를 백그라운드 쓰레드로 조정합니다.

### 3.4 증분 갱신 (`infopilot.py watch`)
- 새 파일 이벤트는 `watchdog` 기반 핸들러가 큐에 적재하고, `IncrementalPipeline`이 증분 임베딩을 수행합니다.
- SentenceTransformer 로딩은 `ModelManager`(아래 5절 참조)를 통해 캐시되며, watcher 종료 시 참조 카운트가 해제됩니다.

## 4. Knowledge & Search Agent (지식·검색 비서)
- CLI 경로: `infopilot.py chat`.
- 준비 단계: `LNPChat.ready()`가 코퍼스·모델·인덱스를 로드하고, 필요 시 번역기(`deep-translator`) 캐시를 초기화합니다 (`core/conversation/lnp_chat.py`).
- 주요 기능
  - 의미 검색: Sentence-BERT 임베딩 기반 유사도, BM25 혼합 (옵션), Cross-Encoder 재랭킹.
  - 정책 적응: `policy_scope`(`auto`/`policy`/`global`)에 따라 검색 범위를 동적으로 제한.
  - 미리보기 번역, 후속 질문 제안, 세션 상태 추적(`SessionState`).
  - 캐시 저장소 구조: `index_cache/doc_meta.json`, `doc_embeddings.npy`, `doc_index.faiss`.
- 모델 관리: `ModelManager`가 SentenceTransformer 인스턴스를 재사용하고 참조 해제를 통해 GPU/CPU 메모리를 절약합니다 (`core/infra/models.py`).
- 예약 실행: `infopilot.py schedule` 명령이 `JobScheduler`(`core/infra/scheduler.py`)를 통해 `indexing.mode = scheduled` 정책에 대한 주기적 스캔·학습을 수행합니다.

## 5. Meeting Agent (회의 비서 MVP)
- 모듈 위치: `core/agents/meeting/`.
- 구성 요소
  - `MeetingPipeline`은 STT(음성 → 텍스트)와 요약 로직을 오케스트레이션하는 플레이스홀더입니다.
  - `MeetingJobConfig`는 오디오 경로, 출력 디렉터리, 언어, 정책 태그 등을 정의합니다.
  - 현재 STT/요약 구현은 자리 표시자이며, Whisper·LLM 등을 연결할 확장 포인트를 남깁니다 (`_transcribe`, `_summarise`).
  - 실행 결과는 `transcript.txt`, `summary.json`, `segments.json`으로 저장되어 요약, 액션 아이템, 결정 사항, 세그먼트 타임라인을 제공합니다.
- 정책 연동: `policy_tag` 필드로 스마트 폴더 정책과 일관성 있게 관리하도록 설계되었습니다.

## 6. Photo Agent (사진 비서 MVP)
- 모듈 위치: `core/agents/photo/`.
- 파이프라인 흐름
  1. 루트 경로 스캔(`_scan`) – JPEG/PNG/HEIC 파일을 탐색해 `PhotoAsset`으로 수집.
  2. 태깅(`_tag`) – 현재는 임시 태그와 임베딩을 채우지만, 비전 백엔드 연동을 위한 인터페이스를 제공합니다.
  3. 중복 판별(`_deduplicate`) – 파일 크기 기반 그룹화 후 2개 이상이면 중복으로 표기.
  4. 베스트샷 추천(`_pick_best`) – 수정 시각 기준 상위 20장까지 선별.
  5. 결과 저장 – `photo_report.json`에 베스트샷, 중복 그룹, 정책 태그를 기록.
- 향후 GPU/ONNX 지원, 유사도 계산 개선 등을 염두에 둔 구조입니다.

## 7. 모델 및 리소스 관리
- `ModelManager`는 문자열 키로 로드된 모델을 캐시하고 참조 카운트를 제공하여 다중 워커 환경에서도 동일 모델을 재사용합니다 (`core/infra/models.py`).
- `JobScheduler`는 `ScheduleSpec`/`ScheduledJob`을 기반으로 간단한 협력형 예약 실행을 수행하며, cron 또는 간격 기반 반복을 지원합니다 (`core/infra/scheduler.py`).
- `Program` 전체는 `docs/cycles/`에서 사이클별 산출물·리스크를 기록하며, Cycle 1에서 스마트 폴더 P0 기능과 스케줄러/모델 매니저가 완성되었습니다 (`docs/cycles/cycle_1.md`).

## 8. CLI 요약
- `scan` → `train` → `chat`이 기본 플로우이며, `pipeline`은 스캔과 학습을 일괄 실행합니다.
- `watch`는 증분 작업, `schedule`은 정책 예약, `chat`은 대화형 검색, `train`은 독립 학습 단계입니다.
- 모든 명령은 `--policy`로 스마트 폴더 정책을 지정하거나 `--policy none`으로 비활성화할 수 있습니다.

## 9. 향후 확장 포인트
- Meeting/Photo 에이전트는 백엔드 교체가 가능하도록 구성돼 있으며, Whisper·Vision Transformer 등의 실제 모델을 연결할 수 있습니다.
- Knowledge & Search는 멀티 모델 레지스트리와 GPU 오프로딩(`core/infra/offload.py`)의 기반이 준비되어 있습니다.
- 정책 스케줄러는 현재 `knowledge_search` 전용이지만, 에이전트 간 파이프라인을 확장할 수 있도록 메타데이터 구조를 포함합니다.

