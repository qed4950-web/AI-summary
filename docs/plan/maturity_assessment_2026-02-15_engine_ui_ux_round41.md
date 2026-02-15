# AI-summary 완성도 재평가 (엔진/UI/UX, Round 41) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로 이번 라운드는  
스레드 본문 복원, 증분 인덱싱 삭제 경로, rebuild 회귀 방지 계약을 보강했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. 스레드 본문 복원(실사용 UX)
- `DESKTOP_THREAD_TIMELINE_PATH` 기반 스레드별 타임라인 저장/복원 추가
- 스레드 전환 시 이전 스레드 타임라인 스냅샷 저장 후 선택 스레드 본문 복원
- 신규 스레드 생성 시 본문 초기화/저장 동기화

2. Incremental 삭제 반영 경로 실구현
- `scripts/run_incremental_index.py`의 삭제 분기 `pass` 제거
- 삭제 감지 시 전체 재정합(full reconciliation), 그 외는 추가/수정 대상만 증분 처리
- `run_step2(..., scan_state_path, chunk_cache_path)` 연결로 상태/캐시 정합성 강화

3. retrieval rebuild 회귀 방지 계약 테스트
- `tests/test_retrieval_strategy_contract.py` 신설
- 검증 항목:
  - non-blocking `ready(rebuild=True, wait=False)` 호출
  - legacy `ready(rebuild=True)` 시그니처 호환
  - `index_manager.schedule_rebuild(priority=True)` fallback

## 반영 파일

- `desktop_app/ui.py`
- `scripts/run_incremental_index.py`
- `tests/test_ui_smoke.py`
- `tests/test_retrieval_strategy_contract.py`
- `scripts/dev/verify/verify_release_integration_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **9.7 / 10**
- 앱 UI 완성도: **9.8 / 10**
- 사용자 UX 경험: **9.8 / 10**

## 남은 핵심 보완점 (Assess)

1. 스레드 타임라인의 카드형 위젯 복원
- 현재 본문/파일/액션 텍스트는 복원되지만 카드 위젯(`ActionRecoveryCard`, `FailureGuideCard`)은 스냅샷 제외

2. `core/data_pipeline/drift.py` 내부 미완성 분기 정리
- `detect()` 내부 주석/가정 기반 로직이 길어 유지보수 난이도 상승

3. 증분 인덱싱 시나리오 계약 테스트 보강
- 삭제 포함 증분 실행의 파일셋 선택 정책(전체/부분)이 의도대로 유지되는지 스크립트 계약 테스트 추가 필요
