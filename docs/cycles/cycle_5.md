## Cycle 5 – P4 (하이브리드/운영 관리)

### 목표
- 클라우드 오프로딩, 감사 로그, 접근 제어 등 운영 기능 강화
- 정책 기반 모델 자동 선택과 모니터링 대시보드 설계
- 배포/CI 파이프라인을 운영 환경 기준으로 정비

- [x] 오프로딩 전략 및 보안 연동을 위한 유틸리티 초안 (`core/infra/offload.py`)
- [x] 감사 로그/모델 선택 유틸리티 기본 구현 (`core/infra/audit.py`, `.../models.py`)
- [x] 하이브리드 설정 템플릿 `config/hybrid.yaml` 추가
- [ ] 대시보드/경보 통합 계획 수립

### 산출물
- 오프로딩/감사/모델 선택 유틸: `core/infra/`
- 하이브리드 설정 템플릿: `config/hybrid.yaml`
- 회귀 테스트: `tests/test_model_selector.py`, `tests/test_audit_logger.py`

### 다음 단계 체크리스트
- Cycle 5 완료 후 전체 운영 시나리오 회고, 릴리스 준비 점검
- 결정 사항/리스크는 `docs/cycles/cycle_5.md`에 기록
