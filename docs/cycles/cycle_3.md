## Cycle 3 – P2 (회의 비서 MVP)

### 목표
- STT 파이프라인을 연결해 오디오 → 텍스트 변환 및 회의 요약/액션 아이템 추출
- 폴더 정책과 연동해 회의 자료 저장/보안 정책을 준수
- 최소 기능의 작업 센터 연동(회의 기록/요약 푸시)과 연계 테스트

### 진행 현황
- [x] STT 모델/서비스 샘플 파이프라인 구조 마련 (`infopilot_core/agents/meeting/`)
- [x] 요약·액션 추출 로직과 정책 연동을 위한 데이터 모델/스토리지 뼈대 정의
- [ ] 회의 로그 저장 포맷 정의 및 인덱싱 파이프라인 통합
- [ ] 작업 센터/알림 경로 프로토타입

#### 2025-09-27 업데이트
- Whisper diarization 정규화를 도입해 STT 세그먼트의 화자 라벨과 타임스탬프를 정제하고 연속 발화를 병합.
- KoBART·Ollama·BitNet으로 확장 가능한 요약 백엔드 팩토리를 추가하고, UI에서 진단 가능한 상태 확인 API를 노출.
- 입력 오디오 지문 기반 캐시를 적용해 동일 회의 반복 실행 시 산출물을 재사용하며, 파이프라인 테스트를 회귀 케이스로 보강.
- MeetingScreen에 Whisper/요약 백엔드 가용성 표시 및 새로고침 버튼을 추가해 사용자 진단을 지원.
- `MEETING_MASK_PII` 옵션으로 이메일/전화번호를 자동 마스킹하고, 다국어 키워드·품질 지표·JSONL 지표까지 저장해 후속 검색·모니터링을 준비.
- 장시간 회의를 위해 `MEETING_STT_CHUNK_SECONDS` 기반 chunk STT 재시도 로직을 추가하고, 실패 시 자동으로 분할 전사를 수행하도록 보완.
- 액션 아이템을 `tasks.json`, 회의 이벤트를 `meeting.ics`, 연동 정보를 `integrations.json`으로 출력해 캘린더/업무 도구에 곧바로 가져갈 수 있도록 구조화했습니다.

### 산출물
- 에이전트 모델/파이프라인: `infopilot_core/agents/meeting/models.py`, `.../pipeline.py`
- 기본 설정 템플릿: `config/meeting_agent.yaml`
- 정책 예시 업데이트: `infopilot_core/data_pipeline/policies/examples/smart_folder_policy_sample.json`
- 회귀 테스트: `tests/test_meeting_pipeline.py`

### 다음 단계 체크리스트
- Cycle 3 완료 후 STT 품질 검증/피드백, Cycle 4(사진 비서) 계획 조정
- 주요 결정/리스크는 `docs/cycles/cycle_3.md`에 기록
