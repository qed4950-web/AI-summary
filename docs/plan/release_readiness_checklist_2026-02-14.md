# AI-summary 릴리스 준비도 체크리스트 (정적) - 2026-02-14

## 요약

- 정적 준비도: **81 / 100**
- 판정: **조건부 진행 가능** (통합 실행 검증 및 릴리스 워크플로우 보강 필요)

## 이번 반영으로 통과한 항목

1. Watch 명령 기본 안정성
- 의존성 미설치 환경에서 `watch`만 명시 오류로 실패하도록 보강
- 감시 경로 정규화/유효성 검증/fail-fast 적용

2. 정책 엔진 처리 안정성
- policy provider 예외/`None` 반환 fallback 고정
- policy 판정 스키마 편차(`tuple`/`dict`/`bool`/문자열) 수용 + fail-closed
- 비정상 이벤트 경로 방어 및 로그 스로틀링

3. Smoke 게이트 일관성
- `Makefile`의 `SMOKE_TESTS` 단일 소스화
- `verify_smoke_gate_contract.py`로 `smoke.yml`-`Makefile` 정합성 정적 검증
- `lint.yml`에서 smoke gate 계약 검증 단계 추가

## 남은 차단/권장 이슈

1. 릴리스 자동화 워크플로우 부재 (차단)
- `.github/workflows/release.yml`이 없어 릴리스 태그/체인지로그/노트 게이트가 수동

2. 통합 회귀 신호 부족 (권장)
- `integration` 마커가 `7`건으로 낮아, 정책/엔진/UI 경계 회귀 탐지력이 제한적

3. UI 동시성 계약 부족 (권장)
- 런처 in-flight 잠금/재초기화 경합 방지 테스트가 부족

## 바로 실행 가능한 후속 순서

1. `release.yml` + 릴리스 메타 검증 스크립트 추가
2. `integration` 마커 테스트를 정책 리로드/desktop recovery 경계 중심으로 확장
3. UI query/reconnect/reinit 상태 가드를 계약 테스트(`test_ui_ux_contract`)로 고정
4. `make lint-debt-domain-refresh` 실행 후 `docs/plan/lint_domain_refresh_summary.md`를 릴리스 검토 문서에 첨부
