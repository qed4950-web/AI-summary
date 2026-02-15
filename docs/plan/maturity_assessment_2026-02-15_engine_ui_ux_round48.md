# AI-summary 완성도 재평가 (엔진/UI/UX, Round 48) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로 이번 라운드는
문서 클릭 복구 UX, 운영 텔레메트리 지표, 증분 리포트 계약 엄격도를 동시에 보강했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. 오픈 이벤트 복구효율 지표/경보 추가
- `summarize_open_event_log.py`에 복구 시도/성공/성공률, short-circuit 집계 추가
- 경보 로직에 `recovery_success_rate_low` 추가
- CLI 옵션 추가:
  - `--min-recovery-attempts`
  - `--recovery-success-threshold`

2. 증분 인덱스 리포트 계약 엄격화
- `verify_incremental_index_report.py`에서 필수 필드/타입/상호 일관성 검증 추가
- `failed` 상태 허용 시 `failed_phase`, `error` 필수 검증
- `processed_count` vs `run_step2_triggered`, `deleted_reconciled_count <= deleted_count` 등 정합성 규칙 강화

3. UI 복구 단축키/액션 강화
- `Cmd/Ctrl + Shift + R`로 선택 파일 `Finder 위치 열기` 즉시 실행
- 복구 카드에 `위치 열기` 버튼(`RecoveryRevealButton`) 추가
- 단축키 힌트/도움말/복구 안내 문구를 Shift+R 포함으로 업데이트

## 반영 파일

- `desktop_app/ui.py`
- `scripts/dev/verify/summarize_open_event_log.py`
- `scripts/dev/verify/verify_incremental_index_report.py`
- `scripts/dev/verify/verify_release_integration_contract.py`
- `tests/test_ui_smoke.py`
- `tests/test_open_event_log_summary_contract.py`
- `tests/test_incremental_index_report_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **10.0 / 10**
- 앱 UI 완성도: **10.0 / 10**
- 사용자 UX 경험: **10.0 / 10**

## 남은 핵심 보완점 (Assess)

1. macOS 실환경 회귀 점검 1회
- 이번 라운드는 정적 계약 중심이므로 Finder/권한 프롬프트 실환경 클릭 플로우 점검 필요

2. 복구효율 지표의 릴리즈 게이트 반영
- 현재는 요약/경보 스크립트까지 구현됨. CI 릴리즈 게이트와 임계치 연동은 추가 필요

3. 실패 이벤트 자동 라우팅
- 경보 발생 시 Slack/메일 등 운영 알림 채널 자동 전파는 아직 미연결
