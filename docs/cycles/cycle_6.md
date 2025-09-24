## Cycle 6 – 검증 및 출시 준비

### 목표
- KPI 측정 자동화 및 대용량/PII 시나리오 회귀 테스트
- 설치 마법사 및 작업 센터 UX 폴리싱
- 문서화/배포 스크립트 정리, 릴리스 노트 초안 작성

- [x] KPI 측정 스크립트 초안 (`scripts/release_prepare.py`) 및 릴리스 체크리스트 작성
- [x] 릴리스 노트 초안 작성 (`docs/release_notes_draft.md`)
- [x] 대용량/PII 테스트 파이프라인 구성 (tests/regression/test_large_corpus.py, test_pii_scrubber.py)
- [x] 설치 마법사 & 작업 센터 UX 개선 메모 (`docs/ux/improvements.md`)
- [x] 릴리스 문서/배포 스크립트 정리 (`docs/release_notes_draft.md`, `scripts/release_prepare.py`)

### 산출물
- KPI 스냅샷 스크립트: `scripts/release_prepare.py`
- 릴리스 체크리스트: `docs/release_checklist.md`
- 릴리스 노트 초안: `docs/release_notes_draft.md`
- 대용량/PII 회귀 테스트: `tests/regression/test_large_corpus.py`, `tests/regression/test_pii_scrubber.py`
- UX 개선 메모: `docs/ux/improvements.md`
- KPI 산출물 템플릿: `artifacts/kpi.json`, `artifacts/kpi_summary.md` (자동 생성)

### 진행 메모
- `scripts/release_prepare.py`가 KPI JSON/Markdown을 동시에 생성하도록 확장되었으며, 데이터/모델 메타 정보를 포함합니다.
- 릴리스 노트는 Cycle 6 범위에 맞춰 정리 완료 (`docs/release_notes_draft.md`).
- 후속 사이클 정의 전, KPI 실측 지표(검색 지연, 정확도) 파이프라인 연결 작업이 필요합니다.

### 다음 단계 체크리스트
- Cycle 6 완료 후 전체 회고, 최종 릴리스 결정
- 주요 결정/리스크는 `docs/cycles/cycle_6.md`에 기록
