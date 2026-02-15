# AI-summary 완성도 재평가 (엔진/UI/UX, Round 39) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로, 문서 검색 후 클릭 시 미오픈 체감 이슈를 중심으로 보강했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. macOS 파일 열기 복구 경로 강화
- `open` 실패 + Preview/Qt/Finder reveal 실패 시, 마지막으로 상위 폴더 `open` 자동 시도
- 성공 시 사용자 메시지: `기본 앱 열기에 실패해 상위 폴더를 열었습니다.`
- 효과: "검색은 되는데 클릭 후 안 열림" 상황에서 fail-closed 대신 fail-soft UX 제공

2. 미오픈 복구 경로 계약 테스트 추가
- `test_launcher_open_local_file_macos_parent_fallback_contract` 추가
- 취소 오류(`작업이 취소되었습니다.`) 연쇄 실패 상황에서 상위 폴더 fallback 성공 동작 검증

3. 정적 릴리스 계약 동기화
- `verify_release_integration_contract.py` 토큰 확장:
  - `_default_status_log_export_path`, `_path_similarity_score`
  - `CSV Files (*.csv)`, `timestamp_utc`, `timestamp_local`
  - `기본 앱 열기에 실패해 상위 폴더를 열었습니다.`
  - 신규 UI 계약 테스트명 2건

## 반영 파일

- `desktop_app/ui.py`
- `tests/test_ui_smoke.py`
- `scripts/dev/verify/verify_release_integration_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **9.4 / 10**
- 앱 UI 완성도: **9.5 / 10**
- 사용자 UX 경험: **9.6 / 10**

## 남은 핵심 보완점 (Assess)

1. 스레드 사이드바 데이터 소스 실연동
- 현재 `LauncherWindow._populate_sidebar_threads()`는 샘플 데이터 기반이며 `show more`도 안내 문구 상태

2. 엔진 계층의 광범위 `pass` 구간 축소
- `core/` 및 일부 `scripts/`에 예외 무시 경로가 넓어 장애 원인 추적성 저하

3. 개인정보 마스킹의 경로별 증거 강화
- UI 마스킹/정책 경로는 존재하지만, 첨부문서 요약 경로별 마스킹 일관성에 대한 계약 증거를 더 늘릴 필요

## 실행 관련

- 요청에 따라 앱 실행/테스트 실행 없이 코드/계약 기준 정적 보강과 평가만 수행했다.
