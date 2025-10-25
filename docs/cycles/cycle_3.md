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

### 산출물
- 에이전트 모델/파이프라인: `infopilot_core/agents/meeting/models.py`, `.../pipeline.py`
- 기본 설정 템플릿: `config/meeting_agent.yaml`
- 정책 예시 업데이트: `infopilot_core/data_pipeline/policies/examples/smart_folder_policy_sample.json`
- 회귀 테스트: `tests/test_meeting_pipeline.py`

### 다음 단계 체크리스트
- Cycle 3 완료 후 STT 품질 검증/피드백, Cycle 4(사진 비서) 계획 조정
- 주요 결정/리스크는 `docs/cycles/cycle_3.md`에 기록
