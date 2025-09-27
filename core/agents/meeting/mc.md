# 회의 비서 MVP 변경 내역

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
- STT 실패 시 `MEETING_STT_CHUNK_SECONDS` 기반으로 오디오를 분할해 재시도하는 Chunk STTfallback을 추가했습니다.
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
- `core.agents.meeting.cli`로 대시보드/큐를 CLI에서 점검하고, `retraining.process_next` 헬퍼로 재학습 파이프라인에 연결할 수 있습니다.
- 온디바이스 LLM 로더, 컨텍스트 스토어, 액션 아이템 로컬 동기화 스텁, 스마트 폴더 ingest CLI를 추가해 6차 제안 준비를 완료했습니다.

## 구현되지 않은 항목
- 실시간 녹음/마이크 입력 경로: 현재는 파일 기반 사이드카 텍스트만 처리하며 UI에서 직접 녹음을 시작할 수 없다.
- 화자 분리 및 타임스탬프 정교화: 휴리스틱으로 일정 간격을 분할할 뿐 실제 diarization 결과를 반영하지 못한다.

## 향후 구현 예정
- STT 백엔드 설정(모델 경로, 디바이스 선택, API 키 등)을 글로벌 설정과 연동하고, 런타임 검증 및 자원 체크를 추가.
- 요약 품질 향상을 위한 LLM 연동 및 하이라이트/액션/결정 추출 규칙 고도화.

## 개선 라운드 제안
### 1차 제안 (반영 완료 가정)
- ✅ Whisper diarization 결과를 세그먼트에 반영하고 화자 라벨/타임스탬프를 정밀화.
- ✅ Ollama·BitNet 등 선택형 요약 백엔드 지원 및 프롬프트 템플릿 고도화.
- ✅ STT/요약 아티팩트를 해시 기반으로 캐싱해 반복 실행 시간 단축.
- ✅ UI에 STT·요약 백엔드 진단 패널을 추가해 설치 가이드 제공.

### 2차 제안 (1차 적용 후)
- ✅ 언어 자동 감지 및 언어별 요약/키워드 세트 스위칭.
- ✅ 자동 품질 측정(ROUGE·LFQA)과 사용자 피드백 수집 루프 구축.
- ✅ 요약/액션 데이터를 벡터 스토어에 적재해 회의 간 교차 검색 지원.
- ✅ Whisper/요약 모델 GPU·CPU 자원 예약을 중앙 관리.

### 3차 제안 (2차 적용 후)
- ✅ 실시간 스트리밍 요약 모드와 종료 후 정밀 요약 자동 전환.
- ✅ PII 마스킹 옵션을 도입해 민감 정보 보호.
- ✅ 장시간 회의를 위한 Chunk STT 및 재시도 메커니즘 구현.
- ✅ 캘린더/업무 도구(Trello, Jira, Notion 등) 연동 어댑터 추가.

### 4차 제안 (3차 적용 후)
- ✅ 사용자 음성 기반 Speaker ID 학습으로 이름 라벨 제공.
- ✅ 회의 맥락 기반 프롬프트 자동 튜닝 컨텍스트 어댑터 도입.
- ✅ 회의 전후 문서를 자동 수집·첨부하는 스마트 패키징.
- ✅ 단계별 체크포인트/재시작을 지원하는 워크플로 엔진 포팅.

### 5차 제안 (4차 적용 후)
- ✅ 스마트폴더 감시 기반 자동 파이프라인 구축: 새 오디오 파일이 들어오면 STT → 전사/요약/analytics 생성 후 context store에 자동 저장.
- ✅ 회의 분석 대시보드(화자 비중, 액션 추적 등) 제공: `/api/meeting/dashboard` + 프론트 버튼으로 요약 지표 시각화.
- ✅ 에이전트 동작 로그를 표준 감사 포맷으로 내보내 거버넌스 대응: JSONL 스키마 버전/이벤트 타입 추가.
- ▫ 자동 모델 재학습 파이프라인: 데이터셋 누적 후 옵션으로 확장 (GPU/클라우드 환경 시 도입).

### 6차 제안 (5차 적용 후)
- ✅ 온디바이스 LLM(GGUF 등) 동적 로딩과 메모리 매핑: 로컬 데이터만으로 프라이버시 강화.
- ✅ Retrieval-Augmented Summarisation(RAG): context store(회의 전사·요약 기반)에서 유사 회의 검색 → summariser 프롬프트 강화.
- ✅ 액션 아이템 로컬 저장: 회의 output 디렉터리에 `action_items.json`을 기본 생성하고 summary 첨부 목록에 연결.
- ▫ 외부 도구 연동(옵션): Trello, Jira, Notion 등 API creds 제공 시, 액션 아이템을 툴과 양방향 동기화.

✅ 변화 포인트

- 5차: “자동 재학습”을 기본이 아닌 옵션으로 뒤로 빼고, 대신 스마트폴더 자동 파이프라인을 전면에 추가.
- 6차: “액션 아이템 처리”를 로컬 JSON 저장이 기본, 외부 연동은 옵션으로 명확히 구분.

### 7차 제안 (6차 적용 후)
- 유사 회의 클러스터링과 메타 요약을 제공하는 다중 회의 분석.
- 사용자 맞춤 요약 톤/스타일 학습 및 자동 적용 옵션 제공.
- 실시간 다국어 자막/번역 기능으로 글로벌 회의 지원.

### 8차 제안 (7차 적용 후)
- 음성 감정 분석으로 회의 분위기 지표를 시각화.
- 액션 아이템 이행 예측 모델로 리스크 알림 제공.
- 프로젝트·조직 단위 권한 분리와 종단간 암호화 지원.

### 9차 제안 (8차 적용 후)
- 추가 개선사항 없음.

## 다음 작업 순서(2025-Q4)
1. ✅ **STT 후처리 고도화**: faster-whisper 출력에 PyKoSpacing(띄어쓰기)과 hanspell(맞춤법) 보정 적용, Whisper 기본 모델을 `small`로 설정.
2. ✅ **요약 엔진 업그레이드**: KoBART 기반 chunked summarisation 파이프라인 도입, 긴 회의록을 2단계 요약 구조로 처리.
3. ✅ **산출물 스키마 확장**: `summary.json`을 메타·요약 구조로 개편하고 화자/타임스탬프 정보를 추가, transcript.json 옵션 제공.
4. (후속) **액션/결정 추출 고도화**: 규칙 기반 강화 후 KoT5 프롬프트 연동 옵션 검토.

## 스프린트 백로그 (다음 단계 제안)
- **실시간 입력 지원** *(Owner: UX 팀 / Target: Sprint 41)* — UI에서 마이크 녹음과 STT 파이프라인을 직접 연결해 파일 업로드 없이 회의를 캡처하도록 구현.
- **Diarization 정교화** *(Owner: STT 팀 / Target: Sprint 41)* — Whisper 구간/라벨을 세그먼트 통합에 반영하고 타임스탬프 보정을 자동화.
- **STT 글로벌 설정 연동** *(Owner: 플랫폼 팀 / Target: Sprint 42)* — 모델 경로, 디바이스, API 키 등을 환경설정과 UI 양쪽에서 관리하고 실행 전 검증 로직 추가.
- **요약 품질 고도화** *(Owner: NLP 팀 / Target: Sprint 42)* — 언어별 프롬프트 템플릿 보완, LLM 백엔드 실험(예: Bigger KoT5) 및 하이라이트/액션 정확도 개선.
- **액션 아이템 저장/연동** *(Owner: Integrations / Target: Sprint 43)* — 회의별 `action_items.json` 기본 생성과 함께 Trello/Jira/Notion 연동 어댑터 구현.
- **회의 분석 대시보드** *(Owner: 데이터 팀 / Target: Sprint 43)* — `analytics/*.json` 데이터를 시각화하는 웹/데스크톱 뷰 추가하고 주요 지표를 요약 화면에 노출.
- **감사 로그 표준화** *(Owner: 플랫폼 팀 / Target: Sprint 44)* — 에이전트 실행 로그를 감사용 포맷으로 내보내고 보존 정책 정의.
- **재학습 파이프라인 옵션화** *(Owner: ML Ops / Target: Sprint 44)* — `training_queue.jsonl`을 소비해 자동 미세조정 작업을 수행하는 배치 잡과 자원 관리 플로우 설계.
- ✅ 재학습 파이프라인 자동화: 큐 통합 + `--max-runs`/metrics 요약 및 `--watch` polling으로 일괄/지속 미세조정 지원.
