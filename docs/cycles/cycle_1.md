## Cycle 1 – P0 (스마트 폴더 + 코어 엔진)

### 목표
- 정책 엔진을 파일 스캐너/인덱서와 연동해 폴더 기반 제어를 구현
- 폴더 보안·인덱싱 주기·보존 정책을 코어 파이프라인에서 적용
- 작업 스케줄러/모델 매니저의 기본 기능 초안 마련

### 진행 현황
- [x] 정책 로더/검증 통합하여 런타임 적용 (`core/data_pipeline/policies/engine.py` → `_load_policy_engine` 경유)
- [x] 정책 기반 인덱서(스캔/증분) 동작 추가 (`infopilot.py` scan/train/watch 경로, watcher 필터 개선)
- [x] 작업 스케줄러·모델 매니저 초기 버전 구현
- [x] 통합 테스트/로깅 보강 및 문서화

### 산출물
- 정책 엔진: `core/data_pipeline/policies/engine.py`
- 정책 로더 확장: `core/data_pipeline/policies/loader/__init__.py`
- 정책 적용 파이프라인 및 스케줄러 CLI: `infopilot.py` (scan/train/pipeline/chat/watch/schedule 경로)
- 작업 스케줄러: `core/infra/scheduler.py`, 모델 매니저 개선 `core/infra/models.py`
- 테스트 보강: `tests/test_policy_engine.py`, `tests/test_scheduler.py`, `tests/test_model_manager.py`

### 다음 단계 체크리스트
- Cycle 1 완료 후 정책 테스트 회고 및 Cycle 2 계획 수립
- 주요 리스크/결정 사항은 `docs/cycles/cycle_1.md`에 누적 기록
