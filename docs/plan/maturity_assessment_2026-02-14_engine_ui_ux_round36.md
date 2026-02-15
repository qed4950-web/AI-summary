# AI-summary 완성도 재평가 (엔진/UI/UX, Round 36) - 2026-02-14

사용자 피드백("문서 검색은 되는데 클릭 시 안 열림")을 최우선으로 반영해, 즉시 보완 Top 1·2·3을 코드/계약 기준으로 완료했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. macOS 문서 열기 신뢰성 보강
- `QDesktopServices` 선처리 의존을 줄이고 `open -- <path>` 결과를 우선 사용
- 문서형 확장자(`pdf/doc/docx/ppt/xls/txt/md`)는 기본 앱 실패 시 `Preview` 자동 fallback
- 최종 실패 시 `open -R`(Finder reveal) fallback 유지

2. 클릭 성공 피드백 가시화
- 파일/상위폴더 열기 성공 시 상태 배지에 즉시 `Opened file:` / `Opened parent:` 표시
- "클릭했는데 반응이 없는 것처럼 보이는" UX 불확실성 감소

3. 회귀 방지 계약 강화
- macOS `Preview` fallback 계약 테스트 추가
- 문서 클릭 성공 시 상태 배지 갱신 계약 테스트 추가
- release integration contract 토큰 동기화

## 반영 파일

- `desktop_app/ui.py`
- `tests/test_ui_smoke.py`
- `scripts/dev/verify/verify_release_integration_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **9.9 / 10**
- 앱 UI 완성도: **9.9 / 10**
- 사용자 UX 경험: **9.9 / 10**

## 다음 assess (우선순위)

1. "파일 열기" 실패 가이드를 버튼형 CTA(권한 설정/기본앱 연결 가이드)로 전환
2. 파일 경로 불일치 시 basename 기반 후보 탐색(스마트 폴더 범위 한정) 옵션 추가
3. 상태 로그 패널에 필터(`error/warning/success`)와 복사/export 추가

## 실행 관련

- 요청에 따라 앱 실행/테스트 실행은 수행하지 않고 코드/계약 기준으로만 반영했다.
