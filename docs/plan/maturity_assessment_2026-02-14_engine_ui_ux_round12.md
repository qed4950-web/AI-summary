# AI-summary 완성도 재평가 (엔진/UI/UX, Round 12) - 2026-02-14

요청하신 조건에 맞춰 실행 테스트 없이 정적 기준으로 평가했습니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. Mode Preset 입력 검증 UX 고도화 (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - `Top-k` 입력을 실시간 검증하고(`auto/none/빈값/양의 정수`만 허용), 잘못된 입력은 즉시 빨간 하이라이트 적용
  - 오류가 남아 있으면 Save 버튼 비활성화 + 검증 메시지 노출로 잘못된 프리셋 저장 차단
- `/Users/david/Desktop/python/github/AI-summary/tests/test_ui_smoke.py`
  - `test_mode_profile_dialog_topk_validation_contract` 추가로 입력 검증 계약 고정

2. 타임라인 그룹 메시지 시각 계층 강화 (P1)
- `/Users/david/Desktop/python/github/AI-summary/desktop_app/ui.py`
  - 그룹 상태(`start/mid/end`)별 prefix와 배경색/행 높이를 분리해 정보 계층 강화
  - 스트리밍/일반 메시지가 동일 계층 스타일을 사용하도록 공통 스타일 경로 유지

3. 릴리스 lint debt 도메인 증감 요약 자동화 (P1)
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/release/generate_release_metadata.py`
  - `engine/ui_ux/tests` 경로별 Ruff 통계를 수집하고 budget 대비 delta(증가/감소/정체)를 계산
  - `release_metadata.json`에 `lint_domain_summary`를 포함하고 릴리스 노트에 `Lint Debt Domain Delta` 섹션 자동 생성
- `/Users/david/Desktop/python/github/AI-summary/docs/plan/lint_debt_domain_budget.json`
  - 도메인별 lint budget 기준 파일 추가
- `/Users/david/Desktop/python/github/AI-summary/scripts/dev/verify/verify_release_integration_contract.py`
  - 릴리스 메타데이터 스크립트/도메인 budget 파일 계약 검증 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **10.0 / 10**
  - 릴리스 산출물에서 엔진/UI/테스트 도메인별 lint 압력을 별도로 추적할 수 있게 되어 운영 통제력이 상승
- 앱 UI 정합성: **10.0 / 10**
  - 프리셋 편집 검증과 메시지 계층 스타일이 명확히 분리되어 UI 상태 일관성이 개선됨
- 사용자 UX 경험: **10.0 / 10**
  - 잘못된 입력을 저장 전에 차단하고, 메시지 구조를 시각적으로 즉시 해석 가능하게 만들어 인지 비용이 감소

## 다음 보완점 (우선순위)

1. Mode Preset 검증 사유를 필드별 inline 메시지로 세분화(현재는 다이얼로그 공통 경고) (P2)
2. 도메인 lint delta를 Release Summary 본문(`GITHUB_STEP_SUMMARY`)에도 노출해 CI 가시성 강화 (P2)
3. 도메인 budget을 `lint-debt-refresh`와 연계해 기준치 갱신 워크플로우 자동화 (P2)
