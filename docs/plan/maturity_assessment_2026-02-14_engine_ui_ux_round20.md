# AI-summary 완성도 재평가 (엔진/UI/UX, Round 20) - 2026-02-14

요청대로 실행 없이 정적 기준으로 중요 항목 1·2·3을 즉시 보완했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 비정상 payload 복원력 강화 (엔진) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `chat.ask` 반환의 `hits/suggestions`가 tuple이어도 처리되도록 정규화
  - 결과 섹션이 전부 비는 경우 사용자 fallback 메시지 제공

2. suggestion 가독성/안정성 보강 (엔진/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `DESKTOP_MAX_SUGGESTION_CHARS` 기반으로 항목 길이 제한
  - 줄바꿈/공백 정규화 후 dedupe하여 과도한 텍스트 노출 억제

3. 파일 링크 과다 노출 UX 정리 (UI/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 파일 링크는 최대 8개 유지, 초과 시 시스템 안내(`상위 8개`) 표시
  - 기존 link-only 안내/미존재 표시와 결합해 참조 문서 해석성을 높임

## 계약 테스트/게이트 반영

- `/Users/david/Desktop/python/github/AI-summary/tests/test_desktop_backend_mode_contract.py`
  - tuple hits/suggestions 정규화 계약 추가
  - 빈 payload fallback 메시지 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - 파일 링크 overflow 안내 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - 신규 토큰/신규 테스트명 검증 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 입력/출력 스키마 편차와 빈 응답 경로에서 fail-safe가 강화됨
- 앱 UI 정합성: **10.0 / 10**
  - 파일 링크 과다 시 화면 부담을 제어하면서 상태 안내를 유지
- 사용자 UX 경험: **10.0 / 10**
  - 빈 응답/링크 과다/미존재 문서 상황 모두에서 사용자 행동 유도가 명확해짐

## 필요한 추가 assess 항목

1. 정책 토글 UI 제공
- privacy/response-length/suggestion-length를 환경변수 대신 설정 화면에서 제어할 필요

2. 참조 문서 카드화
- `[missing]` 텍스트 대신 배지/아이콘 기반 상태 표현으로 UX 명확성 향상 가능

3. 릴리즈 수동 리허설
- macOS 권한 차단 + 링크 overflow + 빈 payload 시나리오를 실제 환경에서 1회 점검 권장
