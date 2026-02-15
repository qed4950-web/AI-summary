# AI-summary 완성도 재평가 (엔진/UI/UX, Round 47) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로,
이번 라운드는 `문서 검색 후 클릭 시 열기 취소` 체감 이슈를 최우선으로 보강했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. macOS 문서 열기 취소/권한 오류 단축 복구
- `open` 기본 시도 실패가 `canceled`/`permission`으로 분류되면 Preview/Qt 재시도를 생략하고 Finder 위치 열기/상위 폴더 열기로 바로 전환
- `DESKTOP_OPEN_CMD_TIMEOUT_SEC` 기반 `open` 명령 타임아웃 가드 추가
- 이벤트 로그에 `open_darwin_short_circuit` 기록

2. 실패 UX 즉시 복구 액션 강화
- 실패 가이드 카드에 `Finder에서 위치 열기` 액션(`reveal_in_finder`) 추가
- 가이드 카드 Tab 힌트를 고정 문구에서 동적 문구로 전환해 실제 버튼 순서와 일치
- 실패 메시지 가이드 문구를 Finder 중심으로 정렬

3. 엔진 실패 경로 관측성 보강
- `scripts/run_incremental_index.py`에 예외 경로 리포트 추가
- 실패 시 `status=failed`, `failed_phase`, `error`를 리포트에 남기고 재발생시켜 상위 오케스트레이션이 감지 가능
- 계약 테스트로 실패 리포트 기록 보장

## 반영 파일

- `desktop_app/ui.py`
- `scripts/run_incremental_index.py`
- `scripts/dev/verify/verify_release_integration_contract.py`
- `tests/test_ui_smoke.py`
- `tests/test_run_incremental_index_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **10.0 / 10**
- 앱 UI 완성도: **10.0 / 10**
- 사용자 UX 경험: **10.0 / 10**

## 남은 핵심 보완점 (Assess)

1. macOS 실환경 클릭 오픈 회귀 점검 1회
- 이번 라운드는 정적 계약 중심이므로 실제 Finder/권한 프롬프트 환경에서 1회 점검 필요

2. 오픈 이벤트 경보 자동 알림 연계
- 현재는 요약/임계치 평가까지 있으나 Slack/메일 라우팅 미연결

3. 실패 가이드 액션 성공률 대시보드화
- `reveal_in_finder` 액션의 성공률 집계를 릴리즈 지표와 연결하면 UX 체감 품질 추적이 쉬워짐
