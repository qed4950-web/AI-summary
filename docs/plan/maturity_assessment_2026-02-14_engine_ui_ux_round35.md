# AI-summary 완성도 재평가 (엔진/UI/UX, Round 35) - 2026-02-14

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로, 실행 없이 코드/계약 레벨에서 다음 3개를 보강했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. 파일 열기 실패 UX 분기 강화
- `LauncherWindow`에 실패 원인 분류(`permission/canceled/association/not_found/generic`) 추가
- 분류별 안내 문구를 타임라인에 자동 출력
- 기존 복구 액션 카드(다시 열기/상위 폴더/경로 복사)는 유지

2. 히스토리 커스텀 datetime 필터 추가
- `History period` 옆에 `Custom from (YYYY-MM-DD HH:MM)` 입력 + `Apply custom` 버튼 추가
- 입력값을 로컬시간으로 해석 후 UTC `absolute:<iso>` 토큰으로 변환해 필터에 적용
- 잘못된 포맷은 즉시 invalid 표시 + 배너 에러로 피드백

3. 상태 배너 최근 이벤트 히스토리 추가
- Settings Hub에 `Status timeline (recent)` 패널 추가
- 최근 상태 이벤트 N개(기본 8개) 누적 표시
- throttle로 억제된 중복 이벤트는 로그 누적되지 않음

## 반영 파일

- `desktop_app/ui.py`
- `tests/test_ui_smoke.py`
- `scripts/dev/verify/verify_release_integration_contract.py`

## 계약/테스트 보강

- `test_settings_hub_history_filter_custom_datetime_contract` 추가
- `test_launcher_file_open_failure_guidance_contract` 추가
- 기존 상태 배너 계약 테스트에 status log 동작 검증 추가
- release integration contract 토큰 목록 동기화

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **9.8 / 10**
- 앱 UI 완성도: **9.9 / 10**
- 사용자 UX 경험: **9.9 / 10**

## 다음 assess (우선순위)

1. 파일 열기 실패 안내를 "원인+권장 액션 버튼"(권한 설정 열기/기본앱 연결 가이드 열기)으로 구조화
2. 커스텀 datetime 입력에 timezone 라벨 및 최근 사용값 quick preset 추가
3. 상태 로그를 필터링(`error/warning/success`)하고 클립보드 내보내기 추가

## 실행 관련

- 요청에 따라 앱 실행/테스트 실행은 수행하지 않고 코드/계약 기준으로만 반영했다.
