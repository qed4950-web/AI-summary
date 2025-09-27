"""6차 개선 제안 설계 메모"""

# 6차 개선 제안 설계 메모

## 1. 온디바이스 LLM 동적 로딩/메모리 매핑
- 목표: whisper/kobart 외 추가 로컬 LLM(GGUF 등)을 필요시 로딩하고, 메모리 매핑 기반으로 빠르게 교체. (Loader 스캐폴딩 구현 완료)
- 요구사항
  - 모델 레지스트리: 모델명 ↔ 파일 경로 ↔ 디바이스 정보(cpu/gpu).
  - 메모리 매핑 옵션: `mmap=True/False`, 로딩 전략(`lazy`, `prefetch`).
  - 리소스 제한 감지: 사용 가능 메모리/VRAM 확인 후 자동 fallback.
- 구현 초안
  - `core/agents/meeting/llm/loader.py` 생성 → 공통 인터페이스(`load_model`, `unload_model`).
  - 설정 키: `MEETING_ONDEVICE_MODEL_PATH`, `MEETING_ONDEVICE_DEVICE`.
  - CLI/환경 변수로 모델 교체 트리거.
- 검증 계획
  - 더미 모델 파일 사용 → 로딩 로그 및 캐시 확인.
  - 리소스 부족 시 placeholder 잡고 경고 로그 출력.

## 2. RAG(외부 지식 참조) 기반 요약
- 목표: 회의 요약 시 최근 회의록, 문서, 위키 등을 참조하여 요약 품질 향상. (Context store/CLI ingest 스캐폴딩 완료)
- 요구사항
  - 문맥 저장 구조: `analytics/context/<meeting_id>.jsonl` 형태로 문서 스니펫 저장.
  - 임베딩 백엔드 선택: 로컬(embed 모델) vs 외부(APIs).
  - 질의 플로우: 요약 전 `retrieve(top_k)` → prompt augmentation.
- 구현 초안
  - `core/agents/meeting/context_store.py` → `add_document`, `search(query)`.
  - 파이프라인 단계 추가: `_collect_context_bundle` 이후 재사용.
  - 설정 키: `MEETING_RAG_ENABLED`, `MEETING_RAG_TOPK`.
- 검증 계획
  - 더미 문서 넣고 검색 결과를 prompt에 결합.
  - 요약 모델 입력 로그로 확인.

## 3. 액션 아이템 책임자/마감 동기화
- 목표: 요약 단계에서 액션 항목에 담당자/마감 정보 없을 시 추론하거나 사용자가 입력하여 외부 도구와 동기화. (로컬 JSON sync 스텁 완료)
- 요구사항
  - `action_items` 구조 확장: `owner`, `due` 필드 추가.
  - 외부 통합 매핑: Trello/Jira/Notion API 필드 대응 표.
  - 인증 전략: API 키/토큰 저장 위치(`MEETING_INTEGRATIONS_CONFIG`).
- 구현 초안
  - `core/agents/meeting/integrations/sync.py` → `sync_action_items(items, provider)`.
  - `meeting.pipeline`에서 action_items 추출 후 sync 호출 옵션.
  - CLI `integrations push --provider trello --meeting <id>` 추후 지원.
- 검증 계획
  - 더미 provider로 JSON 파일에 쓰는 mock 구현 → 로그 확인.
  - 통합 테스트에서 API 호출 stub.

## 공통 TODO
- 설정/환경 변수 스펙 문서화.
- 테스트 전략: 유닛 테스트 + 통합 테스트 대상 식별.
- 단계별 출시: 장치 모델 로딩 → RAG 검색 → 액션/동기화 순으로 점진적 통합.
