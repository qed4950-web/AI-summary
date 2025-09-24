## Cycle 6 – 검증 및 출시 준비

### 목표
- KPI 측정 자동화 및 대용량/PII 시나리오 회귀 테스트
- 설치 마법사 및 작업 센터 UX 폴리싱
- 문서화/배포 스크립트 정리, 릴리스 노트 초안 작성

- [x] KPI 측정 스크립트 초안 (`scripts/release_prepare.py`) 및 릴리스 체크리스트 작성
- [x] 릴리스 노트 초안 작성 (`docs/release_notes_draft.md`)
- [x] 대용량/PII 테스트 파이프라인 구성 (tests/regression/test_large_corpus.py, test_pii_scrubber.py)
- [x] 설치 마법사 & 작업 센터 UX 개선 메모 (`docs/ux/improvements.md`)
- [ ] 릴리스 문서/배포 스크립트 정리

### 산출물
- KPI 스냅샷 스크립트: `scripts/release_prepare.py`
- 릴리스 체크리스트: `docs/release_checklist.md`
- 릴리스 노트 초안: `docs/release_notes_draft.md`
- 대용량/PII 회귀 테스트: `tests/regression/test_large_corpus.py`, `tests/regression/test_pii_scrubber.py`
- UX 개선 메모: `docs/ux/improvements.md`

### 다음 단계 체크리스트
- Cycle 6 완료 후 전체 회고, 최종 릴리스 결정
- 주요 결정/리스크는 `docs/cycles/cycle_6.md`에 기록
