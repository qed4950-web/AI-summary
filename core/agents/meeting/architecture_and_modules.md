# 아키텍처 & 모듈 설계 개요

## 폴더 구조(제안)
```

AI-Desktop-Assistant/
├─ src/
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ indexer.py               # 파일 스캔/텍스트 추출/증분 인덱싱
│  │  ├─ retriever.py             # 벡터/키워드 혼합 검색, 스코프 필터
│  │  ├─ summarizer.py            # 요약/키포인트/액션아이템 공용 유틸
│  │  ├─ policy_engine.py         # 스마트 폴더 정책 적용(로컬/하이브리드/PII 등)
│  │  ├─ job_scheduler.py         # 배치/주기/야간 작업 스케줄
│  │  ├─ model_manager.py         # 모델 다운로드/캐시/버전 관리
│  │  └─ storage/
│  │     ├─ __init__.py
│  │     ├─ vector/
│  │     │  ├─ __init__.py
│  │     │  ├─ faiss_store.py     # 로컬 FAISS 인덱스 어댑터
│  │     │  └─ chroma_store.py    # 로컬 Chroma 인덱스 어댑터
│  │     └─ cache/
│  │        ├─ __init__.py
│  │        └─ index_cache/       # 검색/미리보기 캐시
│  │
│  ├─ agents/
│  │  ├─ meeting/
│  │  │  ├─ __init__.py
│  │  │  ├─ stt/
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ whisper_service.py # Whisper/Faster-Whisper 래퍼
│  │  │  │  └─ audio_utils.py     # 마이크/파일 I/O, 전처리
│  │  │  ├─ summarizer.py         # 회의 요약/하이라이트/결정/액션아이템
│  │  │  ├─ pipeline.py           # 사후/실시간 처리 파이프라인
│  │  │  └─ export.py             # 지정 폴더에 요약본 저장(.txt/.md), 메타 데이터
│  │  ├─ knowledge_search/
│  │  │  ├─ __init__.py
│  │  │  ├─ embedder.py           # 문서 임베딩(예: sentence-transformers)
│  │  │  ├─ retriever.py          # top-k 검색, 근거 연결
│  │  │  └─ qna.py                # LLM 기반 답변 생성(출처 주입)
│  │  └─ photo/
│  │     ├─ __init__.py
│  │     ├─ tagger.py             # 메타/장소/인물 태깅(옵션)
│  │     ├─ dedupe.py             # 중복/유사 탐지
│  │     └─ bestshot.py           # 베스트샷 추천
│  │
│  ├─ ui/
│  │  ├─ app.py                   # 데스크톱/웹 쉘(예: Electron+Flask/Streamlit)
│  │  └─ components/
│  │     ├─ setup_wizard.py
│  │     ├─ folder_inspector.py
│  │     ├─ work_center.py
│  │     └─ search_view.py
│  │
│  ├─ cli/
│  │  ├─ infopilot.py             # scan/train/chat 오케스트레이터
│  │  └─ commands/
│  │     ├─ scan.py
│  │     ├─ train.py
│  │     └─ chat.py
│  │
│  ├─ configs/
│  │  ├─ policies.yaml            # 스마트 폴더 정책(YAML/JSON)
│  │  └─ models.yaml              # 로컬/클라우드 모델 설정
│  │
│  └─ tests/
│     └─ ...                      # pytest 기반
│
├─ data/                           # 생성된 코퍼스/모델(배포 제외)
├─ index_cache/                    # 검색 캐시(배포 제외)
├─ docs/
│  └─ architecture.md
├─ requirements.txt
└─ README.md

```

## 핵심 인터페이스(스케치)
```python

# 인터페이스 스케치 (의존성 분리 목적의 간단한 프로토콜/ABC)
from typing import Protocol, List, Dict, Any, Iterable
from abc import ABC, abstractmethod

class VectorStore(Protocol):
    def add(self, items: Iterable[Dict[str, Any]]) -> None: ...
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]: ...

class Agent(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def enabled(self, policy: Dict[str, Any]) -> bool: ...

class MeetingAgent(Agent):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str: ...
    @abstractmethod
    def summarize(self, text: str) -> Dict[str, Any]: ...  # {'summary':..., 'actions':..., 'decisions':...}
    @abstractmethod
    def export_summary(self, meta: Dict[str, Any], out_dir: str) -> str: ...

class KnowledgeSearchAgent(Agent):
    @abstractmethod
    def index_documents(self, docs: Iterable[Dict[str, Any]]) -> None: ...
    @abstractmethod
    def answer(self, question: str) -> Dict[str, Any]: ...  # {'answer':..., 'sources':[...]}    

```

## 스마트 폴더 정책 예시(YAML)
```yaml

# 예시: 스마트 폴더 정책 (policies.yaml)
folders:
  - path: "D:/Projects/고객사A"
    agents: ["meeting", "knowledge_search"]
    security:
      processing: "local_only"   # local_only | hybrid
      pii_filter: true
    indexing:
      mode: "scheduled"          # realtime | scheduled | manual
      cron: "0 3 * * *"          # 매일 03:00
    retention:
      summary_days: 180
      preview_cache: false

```

- **회의 비서**: Whisper/Faster-Whisper 래퍼를 통해 로컬 STT → PyKoSpacing + hanspell 후처리 → KoBART chunk summarisation(필요 시 2단계 요약) → 액션/결정 규칙 추출 → 지정 폴더에 `.txt/.md` 저장 → 벡터 스토어(FAISS/Chroma)에 증분 업데이트. STT 백엔드는 `MEETING_STT_BACKEND`, 요약 백엔드는 `MEETING_SUMMARY_BACKEND`로 선택한다. Whisper 설정은 `MEETING_STT_MODEL`(기본 `small`), `MEETING_STT_DEVICE`, `MEETING_STT_COMPUTE`, `MEETING_STT_MODEL_DIR`, KoBART 설정은 `MEETING_SUMMARY_MODEL`, `MEETING_SUMMARY_MAXLEN`, `MEETING_SUMMARY_MINLEN`, `MEETING_SUMMARY_CHUNK_CHARS` 환경 변수 또는 MeetingScreen 옵션으로 조정한다.
-   - 산출물: `summary.json` → `meeting_meta`(제목/날짜/화자 리스트) + `summary`(highlights/action_items/decisions/KoBART 요약) + `raw_summary` 텍스트 + 필요 시 `attachments.transcript`. `action_items`/`decisions`는 `{"text", "ref"(HH:MM:SS) }` 구조를 사용해 원 발화 위치를 가리킨다. `MEETING_SAVE_TRANSCRIPT=1`이면 `transcript.json`에 HH:MM:SS 타임스탬프와 `SPEAKER_n` 라벨을 포함한 발화 로그를 함께 기록한다.
- **지식·검색 비서**: 문서/요약 임베딩 → 혼합 검색(top-k) → LLM으로 최종 응답 구성 + 근거 문서 제공.
- **스마트 폴더**: 폴더별 정책으로 처리 모드/보안/인덱싱 주기 제어. 민감 폴더는 원문 캐시 금지, 요약만 저장.
- **모델 매니저**: 로컬 우선(캐시), 필요 시 클라우드 선택. 키/자격증명은 절대 하드코딩 금지(환경변수/OS 키체인 사용).

## 운영 규칙
- `core/agents/meeting/mc.md`에는 언제나 구현이 완료되지 않은 항목과 앞으로 구현할 작업 목록을 최신 상태로 기록한다. 기능을 추가하거나 변경할 때는 해당 문서를 갱신한 뒤 미완료/추가 예정 항목을 명시해야 한다.
