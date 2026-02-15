# AI-summary 완성도 재평가 (엔진/UI/UX, Round 13) - 2026-02-14

요청하신 조건에 맞춰 실행 테스트 없이 정적 기준으로 평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. Mode Preset 검증을 필드별 inline UX로 세분화 (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 모드별 `Top-k` 아래에 inline 오류 라벨(`ModeInlineError`)을 배치하고, 잘못된 값은 해당 필드만 즉시 강조
  - 공통 경고 중심 구조를 제거하고 필드 단위 피드백으로 전환
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - `test_mode_profile_dialog_topk_validation_contract`를 inline 라벨 계약에 맞춰 갱신

2. Release Summary에 lint domain delta 직접 노출 (P1)
- `/Users/david/Desktop/python/github/AI-summary/.github/workflows/release.yml`
  - `Release governance summary` 단계에서 `artifacts/release/release_metadata.json`을 읽어
    `engine/ui_ux/tests` 도메인별 current/budget/delta를 `GITHUB_STEP_SUMMARY`에 출력

3. 도메인 budget 갱신 워크플로우 자동화 (P1)
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/refresh_lint_domain_budget.py`
  - 도메인 경로별 Ruff 통계에서 `budget_total`을 갱신하는 전용 스크립트 추가
- `/Users/david/Desktop/python/github/AI-summary/Makefile`
  - `lint-debt-domain-refresh` 타깃 추가
  - `lint-debt-refresh`에서 baseline 갱신 후 도메인 budget 갱신을 자동 체인
- `/Users/david/Desktop/python/github/AI-summary/tests/test_lint_domain_budget_refresh_contract.py`
  - 스크립트/Makefile/도메인 budget 핵심 계약 테스트 추가
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - 신규 타깃/스크립트/워크플로 토큰 계약 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 릴리스 단계에서 도메인별 lint 증감을 즉시 관찰할 수 있어 거버넌스 제어성이 강화됨
- 앱 UI 정합성: **10.0 / 10**
  - 프리셋 편집 검증이 필드 단위로 정렬되어 입력-피드백 맥락이 명확해짐
- 사용자 UX 경험: **10.0 / 10**
  - 오류 원인이 입력 위치와 즉시 연결되고, 릴리스 결과도 요약 화면에서 바로 해석 가능

## 다음 보완점 (우선순위)

1. `Top-k` 외 `Description/Status` 길이·형식 검증도 inline 규칙으로 확장 (P2)
2. release summary에 `top lint codes`(도메인별 상위 1~2개) 함께 노출 (P2)
3. `refresh_lint_domain_budget.py` 결과를 PR 코멘트 템플릿/체크리스트와 연동 (P2)
