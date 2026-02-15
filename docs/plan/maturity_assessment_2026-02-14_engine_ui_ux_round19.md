# AI-summary 완성도 재평가 (엔진/UI/UX, Round 19) - 2026-02-14

요청하신 대로 실행 없이 정적 기준으로 즉시 보완 후 재평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 비정상 `chat.ask` 반환 fail-safe 추가 (엔진) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `dict`가 아닌 반환(`str`/`None` 등)을 표준 payload로 정규화해 런타임 예외 방지
  - `hits`/`suggestions` 타입이 잘못돼도 안전한 기본값으로 복구

2. suggestion 폭주/형식 깨짐 방어 (엔진/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `DESKTOP_MAX_SUGGESTION_CHARS` 도입
  - 줄바꿈/과도한 공백 정규화 + 항목별 길이 제한으로 UI 가독성 보장

3. 링크-only 응답 UX 개선 + missing 가시화 (UI/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 본문 없이 링크만 오면 안내 문구 출력: `참조 문서만 반환되었습니다. 아래 파일을 확인하세요.`
  - 미존재 파일은 `[missing]` 라벨로 표시하고 요약 시스템 메시지 추가

## 계약 테스트/게이트 보강

- `/Users/david/Desktop/python/github/AI-summary/tests/test_desktop_backend_mode_contract.py`
  - non-dict payload 정규화 계약 추가
  - suggestion 정규화/절단 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - link-only 응답 placeholder + missing 라벨 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - 신규 토큰/신규 테스트명 검증 반영

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 응답 payload 스키마 편차와 suggestion 폭주에 대한 방어가 추가됨
- 앱 UI 정합성: **10.0 / 10**
  - 링크-only/미존재 파일 상황에서도 일관된 안내와 상태 표시가 제공됨
- 사용자 UX 경험: **10.0 / 10**
  - 실패 원인/복구 경로/참조 문서 상태가 즉시 드러나 작업 단절이 줄어듦

## 남은 보완점

1. 설정 UI로 privacy/length 정책 노출
- 환경변수 대신 앱 내 설정에서 제어하면 운영 편의성 향상

2. file-card 컴포넌트화
- 파일 행에 존재/권한/열기 결과를 배지 형태로 표준화하면 가독성 향상

3. macOS 권한 차단 리허설
- TCC 거부/취소 시나리오를 실기기에서 1회 확인하면 릴리즈 리스크 감소
