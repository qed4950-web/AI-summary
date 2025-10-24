# 사진 비서 아키텍처

사진 비서는 `PhotoAgent` → `PhotoPipeline` → TaskGraph 스테이지로 이어지는 단순 3단 구조입니다.

## 구성 요소

| 계층 | 주요 모듈 | 설명 |
| --- | --- | --- |
| 에이전트 래퍼 | `core/agents/photo/agent.py` | 입력 검증, 진행률/취소 이벤트 처리, 결과 포매터 |
| 파이프라인 | `core/agents/photo/pipeline.py` | TaskGraph를 사용해 `scan → analyse → persist` 세 단계를 순차 실행 |
| 데이터 모델 | `core/agents/photo/models.py` | `PhotoJobConfig`, `PhotoAsset`, `PhotoRecommendation` 데이터 구조 정의 |

## TaskGraph 단계

1. **scan**  
   - 대상 루트 폴더를 재귀 탐색해 이미지(`.jpg`, `.jpeg`, `.png`, `.heic`)를 수집합니다.  
   - 누락된 경로는 경고 로그로 처리하고 넘어갑니다.

2. **analyse**  
   - 태깅(`_tag`): 현재는 플레이스홀더 태그와 임베딩을 채웁니다. 향후 비전 모델 연동 지점입니다.  
   - 중복 검출(`_deduplicate`): 파일 크기를 기준으로 그룹화합니다.  
   - 베스트샷 선택(`_pick_best`): mtime 기준 최신 20장을 추천합니다.  
   - 추천 결과는 `PhotoRecommendation` 객체로 `TaskContext`에 저장됩니다.

3. **persist**  
   - `photo_report.json`을 생성하고 베스트샷/중복/유사 그룹을 JSON으로 저장합니다.  
   - 정책 태그 등 추가 메타데이터를 포함합니다.

TaskGraph 실행 중 취소 이벤트가 설정되면 `TaskCancelled`를 발생시켜 파이프라인을 안전하게 중단합니다.

## 산출물

| 파일 | 내용 |
| --- | --- |
| `<output>/photo_report.json` | 베스트샷/중복 그룹/정책 태그 등 정리 결과 |
| `<output>/` | 추가 산출물은 향후 확장(예: 썸네일, CSV) 예정 |

## 확장 포인트

- `_tag` 함수에 비전 API 혹은 자체 모델을 연결하여 태그 정확도를 향상시킬 수 있습니다.
- `_deduplicate`에서 해시/임베딩 기반 중복 탐지를 도입하면 더 정교한 분류가 가능합니다.
- UI/CLI에서 `progress_callback`을 넘기면 `AgentResult.metadata["stages"]`를 통해 단계별 상태를 확인할 수 있습니다.
