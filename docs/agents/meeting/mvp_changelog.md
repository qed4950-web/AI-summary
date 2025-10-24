# 회의 비서 MVP 변경 내역

> 출처: 이전 `core/agents/meeting/mc.md` 정리. 날짜 순으로 최신 기능과 구조 변화를 기록합니다.

## 2025-09-24
- 회의 파이프라인이 사이드카 텍스트/오디오 기반 전사 로딩을 지원하도록 개선했습니다.
- 하이라이트, 액션 아이템, 결정 사항을 휴리스틱으로 추출하고 요약 본문을 구성합니다.
- `summary.json`, `segments.json`, `metadata.json`, `transcript.txt` 등 작업 산출물을 저장하도록 저장 단계 정비했습니다.
- 한국어 키워드를 포함한 액션/결정 감지 규칙을 추가해 로컬 회의 시나리오를 지원합니다.
- 테스트가 실제 전사 문장 예시를 통해 액션 아이템, 요약 텍스트를 검증하도록 업데이트되었습니다.

## 2025-09-25
- `faster-whisper` 기반 STT 백엔드를 추가해 사이드카 텍스트가 없을 때 자동으로 음성을 전사합니다.
- `MeetingPipeline`이 STT 실패 시에도 안전하게 플레이스홀더를 생성하도록 예외 처리를 강화했습니다.
- `requirements.txt`에 `faster-whisper`를 포함시켜 기본 의존성 목록에 반영했습니다.
- MeetingScreen에서 STT 백엔드 선택, Whisper 옵션 설정, 파이프라인 실행·결과 탐색까지 일괄 수행할 수 있는 UI 흐름을 구현했습니다.
- faster-whisper 결과에 PyKoSpacing과 hanspell 후처리를 적용하는 파이프라인을 추가했습니다.
- KoBART 기반 chunk summariser를 도입해 긴 회의록도 2단계 요약으로 처리할 수 있게 했습니다.
- `summary.json`을 `meeting_meta` + `summary` 구조로 재정비하고, `attachments.transcript` 포인터와 선택적 `transcript.json` 생성을 지원합니다.

## 2025-09-26
- `summary.json`의 `action_items`/`decisions` 항목에 HH:MM:SS 타임스탬프(`ref`)를 추가하고, 화자 라벨은 `SPEAKER_n` 형태로 저장합니다.
- `MEETING_SAVE_TRANSCRIPT` 환경 변수를 통해 `transcript.json` 생성 여부를 제어하고, 생성 시 `attachments.transcript`로 경로를 노출합니다.
- MeetingScreen 로그에 KoBART 자동 요약 섹션을 추가해 사용자가 Raw summary를 바로 확인할 수 있게 했습니다.

## 2025-09-27
- faster-whisper diarisation 결과를 정규화해 화자 라벨과 구간 병합을 고도화했습니다.
- 요약 백엔드 팩토리를 도입해 KoBART·Ollama·BitNet 백엔드를 선택적으로 사용할 수 있게 했습니다.
- 오디오 지문(파일 크기/mtime) 기반 캐시를 추가해 동일 입력 재실행 시 산출물을 즉시 재사용합니다.
- MeetingScreen에 STT/요약 백엔드 상태를 표시하고 새로고침할 수 있는 진단 패널을 추가했습니다.
- 언어 자동 감지로 한국어·영어·일본어·중국어에 맞춘 키워드/문구를 적용하고 품질 지표(압축비, 하이라이트 수 등)를 메타데이터에 기록합니다.
- 회의 요약/액션을 검색용 JSONL 인덱스에 저장해 후속 벡터 스토어 통합에 대비하고, 리소스 진단을 통해 GPU 미사용 시 Whisper를 자동으로 CPU 모드로 실행합니다.
- `MEETING_MASK_PII=1` 설정 시 이메일·전화번호를 `[REDACTED_*]` 토큰으로 마스킹해 산출물과 로그에 민감 정보가 남지 않도록 했습니다.
- STT 실패 시 `MEETING_STT_CHUNK_SECONDS` 기반으로 오디오를 분할해 재시도하는 chunk STT fallback을 추가했습니다.
- `tasks.json`·`meeting.ics`·`integrations.json`을 생성해 액션 아이템/결정 사항을 외부 캘린더·업무 도구와 연동할 수 있는 구조화 데이터를 제공합니다.
- ROUGE/LFQA 기반 품질 지표와 피드백 큐(`feedback_queue.jsonl`)를 추가해 사용자 평가 루프를 구축했습니다.
- 스트리밍 회의를 위한 `StreamingMeetingSession`을 도입해 실시간 스냅샷 생성과 종료 후 정밀 요약을 자동화했습니다.
- MeetingScreen UI에서 실시간 세션 시작/발화 입력/마무리를 지원하도록 스트리밍 파이프라인을 연결했습니다.

## 2025-09-28
- 사용자 등록 음성 프로필을 활용해 `speaker_name`을 부여하고 발화자를 실명으로 식별합니다.
- 회의 전·후 문서를 자동으로 수집해 요약 프롬프트에 주입하고 `attachments/context`로 패키징합니다.
- 컨텍스트 어댑터를 추가해 회의 요약 모델 입력에 사전 문맥을 결합합니다.
- `workflow_state.json`과 `checkpoints/`를 통해 단계별 체크포인트 및 재시작을 지원합니다.
- `MEETING_SPEAKER_PROFILE_DIR`, `MEETING_CONTEXT_PRE_DIR`, `MEETING_CONTEXT_POST_DIR` 환경 변수로 신규 기능을 구성할 수 있습니다.

## 2025-09-29
- 회의 분석 지표(`analytics/<meeting>.json`)와 `dashboard.json`을 생성해 발화 비중, 평균 액션 수 등을 축적합니다.
- 재학습 큐(`training_queue.jsonl`)에 회의별 품질 메트릭과 산출물 경로를 기록해 후속 파이프라인과 연동합니다.
- `analytics_index.jsonl`을 통해 회의별 인덱스를 유지하고, `MEETING_ANALYTICS_DIR` 환경 변수로 저장 경로를 오버라이드할 수 있습니다.
- `core.agents.meeting.cli`의 `--watch` 옵션을 통해 실시간 회의를 모니터링하고, 스트리밍 세션 종료 시 자동으로 정밀 요약 단계로 전환합니다.
- 클라이언트/서버 분리 시나리오를 지원하기 위해 `OnDeviceModelLoader`가 로컬/원격 모델을 동적으로 선택하도록 개선했습니다.
- `speaker_id.py`가 화자 교정 큐(`speaker_feedback.jsonl`)를 기록해 재식별에 반영합니다.
- MeetingScreen에서 액션 아이템을 즉시 외부 TODO 도구로 전달할 수 있도록 `integrations/config.json` 템플릿을 추가했습니다.

## 2025-09-30
- 정책 기반 마스킹(PII/Sensitivity) 결과를 `metadata.policy`에 기록해 감사를 단순화했습니다.
- 회의별 로그를 `audit/<meeting>.log` 파일로 저장하고, CLI 결과 요약에 경로를 함께 출력합니다.
- `--translate` 플래그를 추가해 요약 결과를 대상 언어로 후처리하는 옵션을 제공했습니다.
- Whisper large-v3, Groq Mixtral 등 원격 모델을 사용할 때 API 토큰을 `.env`에서 읽어들입니다.
- `MeetingContextAdapter`가 최신 5개 회의의 요약을 자동으로 연결해 회의 간 맥락을 유지합니다.
- `workflow_state.json`이 실패 지점을 기록해 재실행 시 해당 단계부터 재개합니다.
- `StreamingSummarySnapshot`이 UI에 표시될 수 있도록 JSONL 스트림(`snapshots.jsonl`)을 제공합니다.

## 2025-10-01
- 회의 전 메모/의제 입력을 YAML/Markdown으로 받아 자동으로 context bundle로 변환합니다.
- 회의 후 follow-up 메일 초안을 생성해 `followup_email.md`로 저장합니다.
- Slack/Teams Webhook 통합을 추가해 핵심 요약을 실시간으로 공유합니다.
- 회의 녹음이 업로드되는 S3 경로를 감시해 자동으로 파이프라인을 실행하는 모니터링 잡을 추가했습니다.
- 재학습 Runner가 Ollama 기반 LoRA 미세조정을 지원하도록 확장되었습니다.

## 2025-10-02
- `MeetingPipeline`이 GPU/CPU 사용률, 메모리 점유량을 주기적으로 기록하고 `analytics/resources.jsonl`에 저장합니다.
- 요약 결과의 품질에 민감한 고객사를 위해 `MEETING_FORCE_REDACTION=1` 시 PII 마스킹을 강제합니다.
- `MeetingAnalyticsRecorder`가 사용자 피드백 입력(`feedback_queue.jsonl`)을 집계해 우선순위 지표를 계산합니다.

## 2025-10-03
- 실시간 회의 흐름이 안정화되어 CLI/GUI 공통 모듈(`shared_streaming.py`)에 스트리밍 관련 헬퍼를 통합했습니다.
- `core.agents.meeting.integrations`가 Notion 및 Google Tasks 동기화 어댑터를 추가로 지원합니다.
- `MeetingWorkflowEngine`이 실패 마일스톤을 기록해 Retry 대상을 명확히 표시합니다.

## 2025-10-04
- 회의 파이프라인과 UI 연동이 확정되어 QA 보고서 템플릿(`docs/ux/`)과 KPI 스냅샷(`results/eval_summary.json`)을 공유합니다.
- CLI `--dry-run` 모드를 추가해 오디오/전사 파일 검증만 수행할 수 있게 했습니다.
- 고급 설정을 `configs/meeting_agent.yaml`로 외부화했습니다.

## 2025-10-05
- RAG 파이프라인과의 연동을 위해 회의 요약 인덱스를 `data/meeting_index.jsonl`에 저장합니다.
- 재학습 큐가 누적될 경우 자동으로 Slack 알림을 보내도록 Scheduler 연계를 추가했습니다.
- UI에서 Whisper GPU 가용성 테스트 버튼을 제공해 실행 전에 리소스를 확인할 수 있게 했습니다.

---

- **재학습 파이프라인 옵션화** *(Owner: ML Ops / Target: Sprint 44)* — `training_queue.jsonl`을 소비해 자동 미세조정 작업을 수행하는 배치 잡과 자원 관리 플로우 설계.
- ✅ 재학습 파이프라인 자동화: 큐 통합 + `--max-runs`/metrics 요약 및 `--watch` polling으로 일괄/지속 미세조정 지원.
