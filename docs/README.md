# 문서 안내

AI-summary 저장소의 문서는 기능별로 다음과 같이 정리되어 있습니다.

| 디렉터리 | 주요 내용 | 참고 |
| --- | --- | --- |
| `architecture/` | 전체 시스템 구조, 데이터 흐름, 에이전트 구성 | `overview.md` |
| `guides/` | 실사용 가이드 (로컬 LLM 연결, 회의 모델 등) | `local_llm.md`, `meeting_models.md`, `ui_help.md` |
| `process/` | 운영 프로세스와 릴리스 체크리스트 | `release.md` |
| `research/` | 벤치마크, 튜닝 노트, 측정용 스크립트 | `performance_tuning.md`, `benchmarks/` |
| `roadmap/` | 향후 기능 계획 및 우선순위 | `assistant.md` |
| `ux/` | UI 개선 메모 및 시안 | `improvements.md`, `smart_folder_glass_ui.md` |

### 벤치마크 위치

벤치마크 실행 스크립트의 최신 버전은 `scripts/benchmarks/`에 있습니다.  
`docs/research/benchmarks/` 폴더의 `.py` 파일은 과거 실험 재현용으로 보관하며, 문서화된 절차는 `research/performance_tuning.md`에서 최신 경로로 업데이트했습니다.

### 문서 컨벤션

- 모든 문서는 Markdown UTF-8을 사용하며, 코드/명령어는 fenced code block으로 표기합니다.
- CLI 명령은 저장소 루트에서 실행되는 것을 기본으로 설명합니다.
- 문서 수정 시 관련 항목의 상단 목차(해당 README)도 함께 갱신해 주세요.
