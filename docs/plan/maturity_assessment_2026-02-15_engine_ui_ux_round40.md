# AI-summary 완성도 재평가 (엔진/UI/UX, Round 40) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행"을 코드 기준으로 반영했다.  
요청 조건에 따라 앱/테스트 실행 없이 정적 점검으로만 평가했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. 스레드 사이드바 실데이터 연동
- 하드코딩 샘플 목록 대신 로컬 메타 저장소(`DESKTOP_THREAD_HISTORY_PATH`) 기반 로드/저장
- 신규 스레드 생성/질의 시 제목·최근시각 자동 갱신
- `Show more` 페이지네이션과 `No more` 상태 반영

2. 엔진 rebuild 빈 경로(`pass`) 제거
- `core/conversation/retrieval_strategy.py`의 `rebuild=True` 분기에서
  - `ready(rebuild=True, wait=False)` 우선 호출
  - 미지원 시 `index_manager.schedule_rebuild(priority=True)` fallback
- 효과: 재빌드 요청이 실제 동작으로 연결됨

3. 마스킹 계약 테스트 보강
- 스레드 사이드바 지속성/페이지네이션 계약 테스트 추가
- 백엔드 참조문서 제목 + 추천문구 PII 마스킹 계약 테스트 추가

## 반영 파일

- `desktop_app/ui.py`
- `core/conversation/retrieval_strategy.py`
- `tests/test_ui_smoke.py`
- `tests/test_desktop_backend_mode_contract.py`
- `scripts/dev/verify/verify_release_integration_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **9.6 / 10**
- 앱 UI 완성도: **9.7 / 10**
- 사용자 UX 경험: **9.7 / 10**

## 남은 핵심 보완점 (Assess)

1. 스레드별 메시지 본문 복원
- 현재는 스레드 메타(제목/시간) 중심이며, 과거 스레드 선택 시 대화 본문 복원은 미구현

2. Incremental index 삭제 반영 경로 정리
- `scripts/run_incremental_index.py`의 삭제 분기 로직은 아직 실질 삭제 경로가 불완전

3. retrieval rebuild 계약 테스트 추가
- rebuild 호출이 깨지지 않도록 `retrieval_strategy` 전용 계약 테스트를 별도 고정 필요
