## Cycle 0 – 기반 정리 및 준비

### 목표
- 코어 저장소 구조 및 패키지 스켈레톤 정리
- 스마트 폴더 정책 스키마/설정 포맷 초안 마련
- 공용 로깅·테스트 인프라 준비

### 진행 현황
- [x] 패키지 구조 재정비 및 모듈 네임스페이스 검토
- [x] 정책 스키마 초안(예: JSON Schema) 작성 (`core/data_pipeline/policies/schema/smart_folder_policy.schema.json`)
- [x] 테스트/로깅 공통 유틸 초안 작성 (`core/utils/`)
- [x] 의존성/도구 목록 정리 및 환경 스크립트 정의 (`scripts/setup_env.sh`)

### 산출물
- 정책 스키마 예시: `core/data_pipeline/policies/examples/smart_folder_policy_sample.json`
- 공용 유틸 모듈: `core/utils/`
- 환경 셋업 스크립트: `scripts/setup_env.sh`
- 정책 로더/검증 초안: `core/data_pipeline/policies/loader/__init__.py`
- 의존성 업데이트: `requirements.txt`, `requirements_win313.txt`

### 다음 단계 체크리스트
- Cycle 0 완료 후 회고 및 Cycle 1 준비 미팅
- 필요 시 추가 세부 티켓 분배
