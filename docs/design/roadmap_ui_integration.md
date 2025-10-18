# UI Integration Roadmap (2024-11-24)

## 우선순위
1. **LLM 명령 인터페이스 (Toolbar ↔ LNPChat 연결)** *(착수됨, 2024-11-24)*
   - Electron ↔ Python IPC 채널 확장 (`llm-chat` 가칭)
   - 툴바에서 자연어 명령 → LNPChat 응답 패널 표시
   - 대화 히스토리 및 후속 질문 흐름 설계
   - pytest: LNPChat IPC 핸들러의 happy/edge case 케이스 추가

2. **사용자 문서 학습 플로우** *(진행 중, 2024-11-24)*
   - 사용자 선택 폴더 → 학습 큐 → Corpus/Index 갱신
   - 진행 상태 및 완료 알림 UI/IPC
   - pytest: 파일 스케줄링/코퍼스 업데이트 단위 테스트, 엔드투엔드 스모크

3. **스마트 폴더 관리 확장**
   - 스마트 폴더 생성/편집 UI
   - 에이전트별 하위 폴더 지정, 파일 유형(문서/이미지/음성) 매핑
   - `smart_folders.json` 업데이트 및 폴더 스캔 파이프라인 조정
   - pytest: 정책/폴더 구성이 올바로 반영되는지 검증

## 공통 작업 항목
- 각 단계 완료 시 관련 문서(이 파일 포함) 업데이트
- 가능한 범위에서 IPC 단위 테스트 및 Python 쪽 pytest 병행
- 사용자 피드백 반영하며 반복 개선
- 현재 진행 상황
  - 2024-11-24: `chat-llm` IPC 연결 및 Electron UI 반영, `tests/test_chat_pipeline.py`로 기본 JSON 응답 검증 추가

---

## 사용자 학습 플로우 설계 초안 (2024-11-24)

### 학습 옵션
1. **전역 학습 (옵션 A)**
   - 사용자가 설치 시 지정한 루트(또는 OS 전체)를 대상으로 파일 스캔·인덱싱.
   - 스마트 폴더 불필요, 에이전트는 전역 파일에 대해 답변 가능.
   - 필수 항목: 학습 루트 확인, 기본 정책(제외 경로 등) 설정, 장시간 학습에 대한 진행 알림.

2. **스마트 폴더 학습 (옵션 B)**
   - 사용자가 생성한 스마트 폴더 아래의 문서/이미지/음성만 학습.
   - 전역 스캔은 비활성화; 스마트 폴더 편집 UI와 연동.
   - 필요 항목: 스마트 폴더 CRUD, 하위 폴더 선택, 파일 유형 필터, 자동/수동 학습 트리거.

### 파이프라인 구조 (Python)
- `scripts/pipeline/train_agent.py`: 학습 요청을 큐잉하는 CLI 추가 (2024-11-24)
- `scripts/pipeline/train_worker.py`: 큐를 소비해 `infopilot.py pipeline`을 실행하는 워커 (2024-11-24)
- `core/api/training.py`: JSONLines 기반 큐/상태 저장소 마련 (2024-11-24)
  - `train_global`: 루트 경로를 받아 Corpus Builder 실행 *(향후 구현)*
  - `train_folder`: 특정 스마트 폴더 ID/경로만 대상으로 Corpus Builder 실행 *(향후 구현)*
- 두 모드 모두 인덱스 재생성 및 캐시 갱신 필요
- 작업 큐/Scheduler (`core.infra.scheduler`)를 재활용하여 학습 Job 관리
- 진행 상태를 IPC로 전송할 수 있도록 log/이벤트 훅 추가

### Electron ↔ Python IPC 초안
- 신규 채널
  - `training-start-global` → payload: `{ roots: [], exclude: [], policy?: string }`
  - `training-start-folder` → payload: `{ folderId, path, types }`
  - `training-status` → 전체 status 맵을 반환 (폴링 기반)
  - `training:status` (이벤트) → 진행률/상태 업데이트 push
- 에러/완료 시 `training:complete`, `training:error` 이벤트로 UI에 알림

### UI 변화
- 툴바 또는 별도 패널에 “학습 설정” 접근 버튼 추가
- 사용자 플로우
  1. 학습 옵션 선택 (전역 or 스마트 폴더)
  2. 전역: 루트/제외 경로 선택 → 학습 시작
     스마트 폴더: 대상 폴더/파일 유형 선택 → 학습 시작
  3. 진행 상황 표시 (상태 메시지 + log history)
  4. 완료 후 인덱스 갱신 안내, 브리핑/다음 액션 제안

### 추후 작업
- 옵션 A 수행 시 개인정보/대역폭 고려한 경고 문구 및 권한 확인
- 옵션 B 강화: 폴더 감시(watchdog)와 연계해 자동 학습 트리거 지원 검토
- pytest 작성: 학습 IPC 호출 → Mock 파이프라인 실행 → 상태 이벤트 검증
- 문서 업데이트: 사용자 가이드, 정책 설명
