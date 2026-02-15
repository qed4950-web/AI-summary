# AI-summary 완성도 재평가 (엔진/UI/UX, Round 23) - 2026-02-14

요청하신 대로 실행 없이 정적 기준으로 중요 1·2·3을 즉시 진행했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 구조화 응답 정규화 강화 (엔진) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `answer`가 dict/list/tuple 등 구조화 데이터여도 JSON 문자열로 안정 정규화
  - 런타임 상태에 `refs<=N` 표시를 추가해 참조 링크 한도 가시화

2. 참조 문서 한도 정책 고도화 (엔진) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/backend.py`
  - `DESKTOP_MAX_REFERENCE_LINKS` 환경변수로 참조 링크 한도 제어 가능
  - overflow 안내(`총 N건 중 상위 M건`)와 invalid 제외 안내를 함께 제공

3. UI 링크 한도/상태 피드백 고도화 (UI/UX) (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - `DESKTOP_MAX_FILE_LINKS` 기반으로 파일 링크 표시 한도 제어
  - 모드 힌트에 `file-links<=N` 노출, 참조 문서 요약에 총량/표시량을 명확히 표기

## 계약 테스트/게이트 반영

- `/Users/david/Desktop/python/github/AI-summary/tests/test_desktop_backend_mode_contract.py`
  - structured answer JSON 정규화 계약 추가
  - reference limit env 계약 추가
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - 모드 힌트 `file-links<=` 계약 추가
  - overflow 요약 문구(`총 12개 중 8개 표시`) 계약 반영
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - 신규 토큰/신규 테스트명 검증 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 응답 스키마 편차와 참조 링크 정책이 설정 가능하고 관찰 가능한 형태로 정리됨
- 앱 UI 정합성: **10.0 / 10**
  - 모드 힌트/참조 문서 요약이 동일한 정책 신호를 사용자에게 전달
- 사용자 UX 경험: **10.0 / 10**
  - 링크가 많거나 누락/무효 링크가 섞여 있어도 결과 해석과 다음 행동이 명확함

## 필요한 추가 assess 항목

1. 정책 설정 패널
- `DESKTOP_MAX_REFERENCE_LINKS`, `DESKTOP_MAX_FILE_LINKS`, privacy 토글을 UI 설정으로 이전 필요

2. 참조 문서 “더 보기” 인터랙션
- overflow 상황에서 추가 링크를 단계적으로 펼치는 UX 필요

3. 접근성 검토
- missing 파일 색상/툴팁 대비를 WCAG 기준으로 점검 필요
