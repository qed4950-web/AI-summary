# 6차 개선 라운드 계획

회의 비서 6차 개선 목표와 구현/검증 계획을 정리한 노트입니다.

## 1. 온디바이스 LLM 동적 로딩·메모리 매핑
- **목표**: Whisper/KoBART 외에도 GGUF 등 로컬 LLM을 필요 시 로딩하고, 메모리 매핑으로 교체 시간을 단축합니다. (`core/agents/meeting/llm/loader.py`)
- **요구사항**
  - 모델 레지스트리: 모델명 ↔ 파일 경로 ↔ 디바이스 정보(cpu/gpu).
  - 메모리 매핑 옵션: `mmap=True/False`, 로딩 전략(`lazy`, `prefetch`).
  - 리소스 제한 감지: 사용 가능 메모리/VRAM 확인 후 자동 fallback.
- **구현 초안**
  - 공통 인터페이스 `load_model`, `unload_model` 정의.
  - 설정 키: `MEETING_ONDEVICE_MODEL_PATH`, `MEETING_ONDEVICE_DEVICE`.
  - CLI/환경 변수로 모델 교체 트리거.
- **검증 계획**
  - 더미 모델 파일로 로딩 로그와 캐시 확인.
  - 리소스 부족 시 placeholder 생성 및 경고 로그 출력.

## 2. RAG(외부 지식 참조) 기반 요약
- **목표**: 회의 요약 시 최근 회의록/문서를 참고해 품질을 향상합니다. (`core/agents/meeting/context_store.py`)
- **요구사항**
  - 문맥 저장 구조: `analytics/context/<meeting_id>.jsonl`에 문서 스니펫 저장.
  - 임베딩 백엔드 선택: 로컬 임베딩 모델 vs 외부 API.
  - 질의 플로우: 요약 전 `retrieve(top_k)` → prompt augmentation.
- **구현 초안**
  - `MeetingContextStore`에 `add_document`, `search(query)` 구현.
  - 파이프라인 단계 `_collect_context_bundle` 이후 컨텍스트 재사용.
  - 설정 키: `MEETING_RAG_ENABLED`, `MEETING_RAG_TOPK`.
- **검증 계획**
  - 더미 문서를 넣고 검색 결과가 prompt에 반영되는지 확인.
  - 요약 모델 입력 로그로 context 결합 여부 검증.

## 3. 액션 아이템 책임자/마감 동기화
- **목표**: 액션 항목에 `owner`, `due` 필드를 채워 외부 도구와 동기화합니다. (`core/agents/meeting/integrations/`)
- **요구사항**
  - `action_items` 구조 확장: 담당자/마감 정보 포함.
  - 외부 통합 매핑: Trello/Jira/Notion API 필드 대응 표.
  - 인증 전략: `MEETING_INTEGRATIONS_CONFIG`에 API 키/토큰 저장.
- **구현 초안**
  - `integrations.sync_action_items(items, provider)` 구현.
  - `MeetingPipeline`에서 액션 추출 후 sync 호출 옵션.
  - CLI `integrations push --provider trello --meeting <id>` (후속).
- **검증 계획**
  - 더미 provider로 JSON 파일에 기록해 로그 확인.
  - 통합 테스트에서 API 호출 stub로 검증.

## 공통 TODO
- 설정/환경 변수 스펙 문서화.
- 테스트 전략: 유닛 + 통합 테스트 대상 식별.
- 단계별 출시: 온디바이스 모델 로딩 → RAG 검색 → 액션 동기화 순으로 점진 통합.
