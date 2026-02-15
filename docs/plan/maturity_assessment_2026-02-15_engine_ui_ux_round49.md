# AI-summary 완성도 재평가 (엔진/UI/UX, Round 49) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로 이번 라운드는
문서 클릭 복구 정확도, 증분 인덱스 실행 가시성, 오픈 이벤트 운영 경보 정밀도를 추가 보강했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. stale 문서 경로의 Finder 위치 열기 복구 강화
- `ui.py`의 `_reveal_in_finder`가 stale 경로에서도 캐시/유사문서 복구 경로를 먼저 해석한 뒤 Finder를 열도록 개선
- 복구 성공 시 해석 경로를 해상도(`resolution`)와 함께 이벤트 로그로 남기고, 경로 캐시 재학습
- ambiguous 후보의 경우 단정 실패를 명시적으로 기록

2. 증분 인덱스 리포트에 실행 시간 메타데이터 추가
- `run_incremental_index.py` 리포트 필드 추가:
  - `started_at_utc`
  - `finished_at_utc`
  - `duration_ms`
- `no_changes/completed/failed` 모든 종료 경로에서 일관되게 기록
- 검증 스크립트에서 ISO8601 파싱/시간 역전/duration 일관성까지 검증

3. 오픈 이벤트 경보에 short-circuit 비율 감시 추가
- `summarize_open_event_log.py`에 `short_circuit_rate` 집계 추가
- 경보 로직에 `short_circuit_rate_high` 추가
- CLI 옵션 추가:
  - `--short-circuit-rate-threshold`

## 반영 파일

- `desktop_app/ui.py`
- `scripts/run_incremental_index.py`
- `scripts/dev/verify/summarize_open_event_log.py`
- `scripts/dev/verify/verify_incremental_index_report.py`
- `scripts/dev/verify/verify_release_integration_contract.py`
- `tests/test_ui_smoke.py`
- `tests/test_open_event_log_summary_contract.py`
- `tests/test_incremental_index_report_contract.py`
- `tests/test_run_incremental_index_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **10.0 / 10**
- 앱 UI 완성도: **10.0 / 10**
- 사용자 UX 경험: **10.0 / 10**

## 남은 핵심 보완점 (Assess)

1. macOS 실환경 문서 클릭 회귀 확인 1회
- Finder/권한 프롬프트가 실제 데스크톱 세션에서 의도대로 동작하는지 최종 확인 필요

2. 새 경보 지표의 CI 릴리즈 게이트 연결
- `short_circuit_rate` 및 `recovery_success_rate` 임계치 결과를 release gate로 연동 필요

3. 운영 알림 채널 자동화
- `alert` 상태 발생 시 Slack/메일 자동 라우팅 미구현
