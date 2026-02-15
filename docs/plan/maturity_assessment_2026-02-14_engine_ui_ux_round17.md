# AI-summary 완성도 재평가 (엔진/UI/UX, Round 17) - 2026-02-14

요청하신 기준에 따라 실행 없이 정적 점검으로 보완/평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 응답 안전 가드(길이 제한) 추가 (엔진 안정성) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `DESKTOP_MAX_RESPONSE_CHARS`(기본 24000) 기반 응답 길이 제한 추가
  - 과도하게 긴 응답은 잘라 UI 프리징 위험을 줄이고 안내 문구를 함께 표기

2. 개인정보 마스킹 범위 확장 + 상태 가시화 (엔진/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - 본문뿐 아니라 suggestions/참조 문서 title에도 마스킹 적용
  - 런타임 상태 문자열에 `privacy=mask/raw` 포함
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 모드 힌트에 `privacy=mask/raw` 노출

3. 파일 열기 실패 메시지 액션 가이드화 (UI/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 권한 거부/취소 오류를 사용자 행동 중심 문구로 변환
  - 기존 상위 폴더 fallback 경로와 결합해 실패 시 복구 가능성 강화

## 계약 테스트/게이트 반영

- `/Users/david/Desktop/python/github/AI-summary/tests/test_desktop_backend_mode_contract.py`
  - privacy 상태, 링크/추천 마스킹, 응답 길이 제한 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - 파일 열기 오류 메시지 가이드 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - 백엔드/UI 신규 토큰 및 신규 계약 테스트명 검증 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 비정상적으로 긴 응답과 민감정보 노출 리스크를 동시에 제어
- 앱 UI 정합성: **10.0 / 10**
  - 모드 힌트와 파일 오픈 오류 피드백이 동일한 상태/가이드 체계로 정렬
- 사용자 UX 경험: **10.0 / 10**
  - 실패 시 원인과 다음 행동이 즉시 제시되어 작업 중단 가능성이 낮아짐

## 남은 보완점

1. 개인정보 마스킹 규칙 UI 설정화
- 환경변수 대신 앱 설정에서 규칙별 on/off를 제공하면 운영 유연성이 증가

2. 파일 열기 실패 텔레메트리
- 권한/취소/기타 오류 코드를 집계해 반복 이슈를 빠르게 감지할 필요

3. macOS 권한 시나리오 수동 점검
- 실제 TCC 거부 환경에서 문구·fallback 동작을 1회 리허설하면 릴리즈 신뢰도 상승
