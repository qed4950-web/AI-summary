# 구성 파일 개요

프로젝트에서 사용하는 구성 파일과 결과 산출물 폴더의 역할을 정리했습니다.

| 위치 | 용도 | 관련 문서 |
| --- | --- | --- |
| `configs/` | 벤치마크·평가 실행 시 사용하는 예시 설정 (`eval_retrieval.json` 등). 스크립트에서 직접 로드하는 실행 프리셋입니다. | `scripts/benchmarks/evaluate_retrieval.py` |
| `core/config/` | 에이전트 런타임 기본 설정 (`meeting_agent.yaml`, `photo_agent.yaml`, `smart_folders.json`, `paths.py`). 코드가 참조하는 프로덕션 기본값입니다. | `docs/agents/*/architecture.md` |
| `results/` | QA/벤치마크 결과 요약을 저장하는 디렉터리 (`eval_summary.json` 등). 배포 전 성능 지표를 남기는 용도입니다. | `docs/agents/meeting/mvp_changelog.md` |

> 실행 프리셋은 `configs/`에, 런타임 기본 설정은 `core/config/`에 위치시켜 구분합니다. 평가 산출물은 `results/`에 저장해 재현성과 이력을 유지하세요.
