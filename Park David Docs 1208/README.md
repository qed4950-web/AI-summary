# Park David Docs 1208 (참고 문서)

이 폴더의 문서들은 AI-summary의 설계/규칙/스켈레톤을 빠르게 공유하기 위한 참고 자료입니다.

## 현재 레포 기준 “정답 경로” (중요)
- Smart Folder 설정: `core/config/smart_folders.json`
- 정책 엔진: `core/data_pipeline/policies/engine.py` (`PolicyEngine`)
- 주요 실행 진입점(파이프라인 CLI): `scripts/pipeline/infopilot.py`

## 주의
- 일부 문서에는 과거 설계안 기준으로 `configs/`(현재는 평가/실험 설정), `core/policy/` 같은 경로 예시가 등장할 수 있습니다. 현재 구현의 정답 경로는 위 목록을 따릅니다.
- 실제 구현/테스트는 현재 레포 구조를 기준으로 하고, 경로가 헷갈리면 위 “정답 경로”를 우선합니다.
