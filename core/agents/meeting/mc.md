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

## 구현되지 않은 항목
- 실시간 녹음/마이크 입력 경로: 현재는 파일 기반 사이드카 텍스트만 처리하며 UI에서 직접 녹음을 시작할 수 없다.
- 화자 분리 및 타임스탬프 정교화: 휴리스틱으로 일정 간격을 분할할 뿐 실제 diarization 결과를 반영하지 못한다.

## 향후 구현 예정
- STT 백엔드 설정(모델 경로, 디바이스 선택, API 키 등)을 글로벌 설정과 연동하고, 런타임 검증 및 자원 체크를 추가.
- 요약 품질 향상을 위한 LLM 연동 및 하이라이트/액션/결정 추출 규칙 고도화.

## 다음 작업 순서(2025-Q4)
1. ✅ **STT 후처리 고도화**: faster-whisper 출력에 PyKoSpacing(띄어쓰기)과 hanspell(맞춤법) 보정 적용, Whisper 기본 모델을 `small`로 설정.
2. ✅ **요약 엔진 업그레이드**: KoBART 기반 chunked summarisation 파이프라인 도입, 긴 회의록을 2단계 요약 구조로 처리.
3. ✅ **산출물 스키마 확장**: `summary.json`을 메타·요약 구조로 개편하고 화자/타임스탬프 정보를 추가, transcript.json 옵션 제공.
4. (후속) **액션/결정 추출 고도화**: 규칙 기반 강화 후 KoT5 프롬프트 연동 옵션 검토.
