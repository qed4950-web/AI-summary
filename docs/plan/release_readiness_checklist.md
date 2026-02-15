# AI-summary 릴리스 준비도 체크리스트

## 요약

- 정적 준비도 기준으로 릴리스 차단 이슈를 먼저 확인한다.
- 릴리스 후보 태그는 `vX.Y.Z` 형식을 사용한다.
- 도메인 lint 갱신 요약은 `docs/plan/lint_domain_refresh_summary.md`에서 확인한다.

## 남은 차단/권장 이슈

1. 차단: 릴리스 게이트(`release.yml`) 실패 항목 해소
2. 차단: 스모크/통합 계약 검증(`verify_smoke_gate_contract.py`, `verify_release_integration_contract.py`) 통과
3. 권장: UI/UX 회귀 체크리스트(검색/단축키/스레드 전환) 수동 점검 기록
4. 권장: `make lint-debt-domain-refresh` 수행 후 `docs/plan/lint_domain_refresh_summary.md` 최신화
