# 회의 비서 아키텍처

회의 비서는 음성(STT) → 텍스트 정규화 → 요약/액션 추출 → 산출물 패키징까지 한 번에 수행하는 파이프라인입니다. 모든 진입점은 `core/agents/meeting` 모듈 집합을 사용하며, CLI/데스크톱 UI는 동일한 API를 호출합니다.

## 상위 구성

| 계층 | 주요 모듈 | 설명 |
| --- | --- | --- |
| 워크플로 제어 | `pipeline.py`, `workflow.py` | 파이프라인 단계(TaskGraph) 실행, 재시작·취소 처리, 체크포인트 관리 |
| 작업 정의 | `agent.py`, `models.py` | 파이프라인 설정(`MeetingJobConfig`), 결과 구조(`MeetingSummary`, `StreamingSummarySnapshot`) 선언 |
| 전사(STT) | `stt/` | Whisper/Faster-Whisper 및 외부 STT 백엔드를 래핑하고, chunk 분할·재시도 로직을 제공 |
| 요약/QA | `summarizer.py`, `llm/` | KoBART·Ollama 등 요약 백엔드 생성, 고정 프롬프트 관리, 후처리 |
| 컨텍스트 | `context_store.py`, `context_adapter.py` | 회의 전/후 문맥, 사용자 메모, 정책 기반 추가 입력 수집 |
| 화자 식별 | `speaker_id.py` | 화자 프로필 로딩, diarisation 결과 정규화 |
| 결과 분석 | `analytics.py` | 발화 비중, 품질 지표, 재학습 큐 등 부가 산출물 기록 |
| 감시/감사 | `audit.py` | 단계별 로그, 실패/성공 이력, 정책 준수 여부 기록 |
| 통합 | `integrations/` | 캘린더·업무 도구 연동을 위한 액션 아이템 내보내기, 공급자 설정 로더 |

## 데이터 흐름 개요

1. **입력 준비**  
   `MeetingJobConfig`가 오디오/전사 경로, 언어, 요약 옵션을 정의합니다. 필요 시 `.env` 혹은 CLI 플래그로 덮어쓸 수 있습니다.
2. **전사 단계**  
   `create_stt_backend()`이 환경 설정에 맞는 STT 백엔드를 선택합니다. 실패 시 chunk 재시도와 캐시 검사를 거쳐 `MeetingTranscriptionResult`를 생성합니다.
3. **요약 및 액션 추출**  
   `create_summary_backend()`가 선택된 요약 엔진을 초기화하고, `MeetingWorkflowEngine`이 하이라이트/결정/액션 아이템 후처리를 수행합니다.
4. **산출물 패키징**  
   `MeetingAnalyticsRecorder`와 `MeetingAuditLogger`가 `summary.json`, `segments.json`, `analytics/*.json`, `feedback_queue.jsonl` 등을 작성합니다.  
   설정에 따라 `transcript.json`, `tasks.json`, `meeting.ics`, `integrations.json` 등이 추가됩니다.
5. **통합 및 재학습**  
   `integrations.sync_action_items()`가 외부 도구로 액션을 내보내고, `retraining*.py` 모듈이 재학습 큐(`training_queue.jsonl`)를 작성합니다.

## 산출물 위치

| 파일 | 용도 | 생성자 |
| --- | --- | --- |
| `summary.json` | 요약 본문 + 하이라이트/액션/결정 | `pipeline.MeetingPipeline` |
| `segments.json` | 타임라인 구간 및 화자 정보 | STT 모듈 |
| `metadata.json` | 입력/환경/성공 여부 기록 | `audit.MeetingAuditLogger` |
| `analytics/<meeting>.json` | 품질 지표, 발화 통계 | `analytics.MeetingAnalyticsRecorder` |
| `training_queue.jsonl` | 재학습 후보 큐 | `retraining_runner.py` |
| `workflow_state.json`, `checkpoints/` | 중간 상태 저장 | `workflow.MeetingWorkflowEngine` |

## 관련 문서

- 실행 가이드: `runbook.md`
- 비용 가이드: `cost_hedging_strategy.md`
- 변경 이력: `mvp_changelog.md`
- 반복 계획: `iteration_plan_round6.md`
