# Release Notes – Cycle 6

## 개요
- 스마트 폴더 정책을 중심으로 한 **지식·검색 비서(InfoPilot Chat)** 워크플로를 정비하고, CLI·UI 모두에서 동일한 검색 품질을 제공합니다.
- 회의·사진 에이전트의 MVP 파이프라인을 구축해 향후 STT/비전 백엔드 연결을 위한 토대를 마련했습니다.
- 하이브리드 운영을 대비한 오프로딩 훅, 감사/권한 스텁, KPI 스냅샷 자동화 스크립트를 정리했습니다.

## 주요 개선 사항
- `infopilot.py chat`
  - `LNPChat`이 Cross-Encoder 재랭킹, 정책 스코프, 다국어 번역·미리보기 캐시, 후속 질문 제안을 통합했습니다.
  - `--scope auto|policy|global`, `--lexical-weight`, `--show-translation` 등 주요 플래그를 문서화했습니다.
- 데이터 파이프라인
  - `scan → train → chat` 자동화 루틴이 정책 엔진과 연결되어 최신 스캔 결과만 학습하도록 보정되었습니다.
  - 증분 모드(`watch`, `schedule`)가 정책 기반 폴더 선택과 예약 파이프라인을 지원합니다.
- 회의/사진 에이전트
  - `core/agents/meeting`과 `core/agents/photo`에 파이프라인 스켈레톤 및 결과 산출 구조를 정리했습니다.
  - 결과 리포트(`summary.json`, `photo_report.json`) 형식을 문서화해 후속 백엔드 연동 시 바로 재사용 가능합니다.
- 운영 툴링
  - `scripts/release_prepare.py`가 KPI 스냅샷과 인공 지표 요약(`artifacts/kpi.json`, `kpi_summary.md`)을 생성하도록 개선되었습니다.
  - 릴리스 체크리스트(`docs/release_checklist.md`)와 사이클 로그에 업데이트 항목을 반영했습니다.

## 품질 & 테스트
- 스마트 폴더 정책 회귀 테스트와 대용량/PII 회귀 테스트(`tests/regression/test_large_corpus.py`, `test_pii_scrubber.py`)를 정비했습니다.
- Meeting/Photo 파이프라인에 대한 연기(smoke) 시나리오와 작업 센터 UX 체크리스트 초안을 마련했습니다 (`docs/ux/improvements.md`).
- KPI 스냅샷 스크립트가 산출물을 생성하는지 수동 검증했습니다 (`python scripts/release_prepare.py --print`).

## 배포 & 운영 메모
- 릴리스 준비 절차는 `docs/release_checklist.md`를 따라 `scan → train → chat` 스모크 테스트 후 KPI 스냅샷을 캡처합니다.
- `artifacts/kpi_summary.md`는 배포 메일/공유 문서에 바로 붙일 수 있는 간단한 메트릭 템플릿을 제공합니다.
- `docs/cycles/cycle_6.md`에 Cycle 6 결정 사항과 TODO 정리가 완료되었습니다.

## 알려진 제한 사항
- Meeting Agent는 Whisper/STT 백엔드와의 실제 연동이 남아 있으며, Photo Agent 역시 비전 임베딩·GPU 경로가 자리 표시자 상태입니다.
- KPI 스냅샷은 아직 실제 지표(검색 지연, 정확도 등)를 수집하지 않으므로, 운영 환경에서 텔레메트리/벤치마크 파이프라인을 연결해야 합니다.
- 설치 마법사/작업 센터 UX는 추가 폴리싱 체크리스트가 남아 있습니다 (`docs/ux/improvements.md`).

## 업그레이드 가이드
1. `python infopilot.py pipeline --out data/found_files.csv`로 최신 코퍼스와 모델을 준비합니다.
2. `python infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache index_cache`로 챗봇 동작을 검증합니다.
3. `python scripts/release_prepare.py`를 실행해 KPI 산출물을 생성하고, 결과를 릴리스 노트와 함께 공유합니다.
