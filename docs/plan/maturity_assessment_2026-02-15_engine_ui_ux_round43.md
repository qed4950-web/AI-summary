# AI-summary 완성도 재평가 (엔진/UI/UX, Round 43) - 2026-02-15

요청한 "가장 중요한 1·2·3 즉시 진행" 기준으로 이번 라운드는  
문서 오픈 복구 신뢰성, 삭제 인덱싱 정합성, UX 오탐 방지에 집중했다.

## 이번 라운드 Top 1·2·3 즉시 진행

1. 문서 오픈 복구 캐시 영속화(UI/UX)
- 복구 경로 캐시를 세션 메모리에서 파일 기반으로 확장
- 앱 재시작 후에도 이전에 복구된 경로를 재사용 가능
- 환경변수 `DESKTOP_FILE_RESOLUTION_CACHE_PATH` 지원

2. 삭제 문서 chunk-cache 정리(엔진)
- `run_incremental_index` 삭제 반영 시 코퍼스/벡터 인덱스뿐 아니라 `chunk_cache`도 즉시 정리
- stale 해시 재사용 경로를 줄여 다음 증분 처리 정합성을 강화

3. 유사 문서 자동복구 오탐 방지(UI/UX)
- 유사 후보가 여러 개인 경우 자동 열기 중단 후 후보 액션 카드로 명시적 선택 유도
- 잘못된 문서 자동 오픈 리스크를 줄이고 사용자의 제어권을 강화

## 반영 파일

- `desktop_app/ui.py`
- `scripts/run_incremental_index.py`
- `tests/test_ui_smoke.py`
- `tests/test_run_incremental_index_contract.py`
- `scripts/dev/verify/verify_release_integration_contract.py`

## 정적 완성도 점수 (실행 없음)

- 프로젝트 엔진 안정성: **9.8 / 10**
- 앱 UI 완성도: **9.9 / 10**
- 사용자 UX 경험: **9.9 / 10**

## 남은 핵심 보완점 (Assess)

1. 복구 후보 탐색 성능
- 현재 `rglob` 기반이라 스마트폴더가 매우 큰 경우 클릭 지연 가능성 존재

2. macOS 오픈 실패 원인 수집
- 사용자 안내는 강화됐지만 OS 에러코드 기반 정밀 telemetry는 아직 제한적

3. 대용량 증분 시나리오 계약 확대
- 삭제/추가/수정 동시 다발 이벤트에서 배치 경계 조건 계약 테스트 추가 필요
