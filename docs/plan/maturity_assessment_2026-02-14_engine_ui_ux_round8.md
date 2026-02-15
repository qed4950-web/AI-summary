# AI-summary 완성도 재평가 (엔진/UI/UX, Round 8) - 2026-02-14

이 문서는 요청하신 조건에 맞춰 실행 테스트 없이 정적 기준으로 평가한 결과입니다.

## 이번 라운드 즉시 진행한 중요 1·2·3

1. 엔진 fail-closed 강화 (P1)
- `scripts/pipeline/infopilot_cli/pipeline_runner.py`
  - `require_policy_engine` 옵션 추가
  - 정책 엔진 불가용 시 add 이벤트를 차단하는 `_filter_add_paths_for_policy_gate` 추가
- `scripts/pipeline/infopilot_cli/watch.py`
  - 정책 경로가 설정된 watch 실행에서는 `require_policy_engine=True` 강제
- `tests/test_pipeline_policy_provider_contract.py`
  - 정책 엔진 필수 모드에서 차단/허용 계약 테스트 추가

2. 앱 UI 레퍼런스 반영 (P1)
- `desktop_app/ui.py`
  - 컴포저 하단에 `Auto/Instant/Thinking/Pro` 모드 드롭다운 추가
  - 모델 칩에 모드 상태 연동 표기
  - 레퍼런스형 다크 컴포저에 맞춘 모드 버튼 스타일 추가

3. UX 조작성/발견성 강화 (P2)
- `desktop_app/ui.py`
  - 컴포저 하단 단축키 힌트 바 추가 (`검색/입력/타임라인/인용/열기`)
  - `Cmd/Ctrl+M` 모드 순환 단축키 추가
  - 단축키 도움말 버튼 추가
- `tests/test_ui_smoke.py`
  - 모드 버튼 기본값 및 모드 순환 계약 검증 추가

## 프로젝트 거버넌스 보완

- `scripts/dev/release/generate_release_metadata.py`
  - `Impact Score` 계산을 외부 정책 파일 기반으로 전환
  - 릴리즈 노트에 `Impact Score Policy` 섹션 추가
- `docs/plan/impact_score_policy.json`
  - 점수 가중치/상한/티어 임계값 정책 분리
- `scripts/dev/verify/verify_release_integration_contract.py`
  - 정책 파일 존재/형식 및 release metadata 스크립트 토큰 계약 검증 추가

## 완성도 점수 (정적 평가, 10점)

- 프로젝트 엔진 안정성: **9.8 / 10**
  - 정책 엔진 불가용 시 watcher add-path fail-open 리스크를 fail-closed로 전환
- 앱 UI 정합성: **9.7 / 10**
  - 레퍼런스 화면의 모드 선택 흐름을 컴포저에 직접 반영
- 사용자 UX 경험: **9.6 / 10**
  - 단축키 발견성과 조작 피드백(힌트/모드 상태)이 강화됨

## 다음 보완점 (우선순위)

1. `core/` 대규모 lint debt(`W293/F401`)를 모듈 단위로 상계 감축 (P1)
2. 모드 드롭다운 선택값을 실제 백엔드 추론 프로필과 연동 (P1)
3. 스레드/타임라인에 hover 상태 및 메시지 그룹 시각 구분 추가 (P2)
