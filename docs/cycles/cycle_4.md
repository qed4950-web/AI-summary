## Cycle 4 – P3 (사진 비서 MVP)

### 목표
- 이미지 태깅, 중복/유사 탐지, 베스트샷 추천 파이프라인을 구현
- GPU/CPU 선택 정책 및 캐시 관리 전략 확립
- 스마트 폴더 정책과 연동해 사진 인덱싱 스코프/보안 설정을 적용

- [x] 이미지 태깅/임베딩 파이프라인 골격 및 설정 템플릿 추가 (`core/agents/photo/`, `config/photo_agent.yaml`)
- [x] 중복/유사 사진 감지를 위한 기본 그룹핑 로직 프로토타입 구현
- [ ] 베스트샷 추천 목적 함수/평가 지표 정의
- [ ] 정책 기반 사진 인덱서/캐시 관리 설계

### 산출물
- 사진 에이전트 모델/파이프라인: `core/agents/photo/models.py`, `.../pipeline.py`
- 설정 템플릿: `config/photo_agent.yaml`
- 정책 예시 업데이트: `core/data_pipeline/policies/examples/smart_folder_policy_sample.json`
- 회귀 테스트: `tests/test_photo_pipeline.py`

#### 2025-10-17 점검
- 베스트샷 추천 목적 함수와 품질 지표가 `core/agents/photo/pipeline.py`에 플레이스홀더로 남아 있어, GPU/ONNX 백엔드 성능 평가와 함께 정의가 필요합니다.
- 정책 기반 사진 인덱서 및 캐시 관리 설계를 위해 스마트 폴더 정책(`core/data_pipeline/policies/engine.py`)과의 연동 시나리오를 작성해야 합니다.

### 다음 단계 체크리스트
- Work Center/Meeting/Operations 우선 과제가 마무리된 이후 베스트샷 목적 함수(품질 지표 + 정책 가중치) 정의 및 샘플 데이터 벤치마크 실행
- 스마트 폴더 정책 기반 캐시 전략을 `core/data_pipeline/policies/engine.py`와 연동하는 설계 문서 작성
- Cycle 4 완료 후 사진 파이프라인 성능 검증, Cycle 5(하이브리드/운영) 준비
- 결정 사항/리스크는 `docs/cycles/cycle_4.md`에 기록
