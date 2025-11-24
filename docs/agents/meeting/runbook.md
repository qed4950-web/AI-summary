# 회의 비서 실행 방법 (Runbook)

로컬에서 회의 비서 파이프라인을 검증하는 최소 절차입니다.

## 1. 환경 준비

```bash
source ./conda/bin/activate ai-summary
# 새 환경이 없다면
# ./conda/bin/conda create -n ai-summary python=3.11 -y
# source ./conda/bin/activate ai-summary
pip install -r requirements.txt
```

환경 변수는 `.env` 혹은 CLI 플래그로 지정할 수 있습니다. 예시는 `docs/guides/meeting_models.md`를 참고하세요.

## 2. 단일 테스트 실행

```bash
python -m pytest tests/test_meeting_dataset_loader.py
```

STT/요약 파이프라인 전체를 검증하고 싶다면 오디오 샘플을 준비한 뒤 다음 명령을 사용할 수 있습니다.

```bash
python scripts/run_meeting_agent.py \
  --audio path/to/sample.m4a \
  --output-dir data/meeting_outputs/sample \
  --language ko
```

## 3. 자주 쓰는 환경 변수

| 변수 | 설명 |
| --- | --- |
| `MEETING_STT_BACKEND` | `wav2vec2`, `whisper` 등 STT 엔진 선택 |
| `MEETING_SUMMARY_BACKEND` | `kobart`, `ollama`, `openai` 등 요약 엔진 선택 |
| `MEETING_SUMMARY_REVIEW_BACKEND` | `ollama` 등 검수 LLM을 지정하면 1차 요약을 재검토 |
| `MEETING_SUPERVISOR_BACKEND` | `ollama` 등 결과 감독 LLM 지정 |
| `MEETING_SUPERVISOR_MODE` | `auto`/`always`/`manual`/`off` 중 선택 |
| `MEETING_SUMMARY_REVIEW_MODE` | `auto`(기본), `always`, `manual`, `off` 중 선택 |
| `MEETING_MASK_PII` | `1`이면 이메일/전화번호를 `[REDACTED_*]` 토큰으로 마스킹 |
| `MEETING_SAVE_TRANSCRIPT` | `1`이면 `transcript.json`을 생성 |

Wav2Vec2 백엔드를 사용할 때 chunk/stride 길이를 조정하려면 `MEETING_WAV2VEC2_CHUNK`, `MEETING_WAV2VEC2_STRIDE` 값을 초 단위로 지정하세요.

권장 기본 설정 예시:

```bash
export MEETING_SUMMARY_BACKEND=ollama
export MEETING_SUMMARY_OLLAMA_MODEL=eeve_korean_v2
export MEETING_SUMMARY_REVIEW_BACKEND=ollama
export MEETING_SUMMARY_REVIEW_MODE=auto
export MEETING_SUPERVISOR_BACKEND=ollama
export MEETING_SUPERVISOR_MODE=auto
export SUMMARY_SUPERVISOR_MODEL=eeve_korean_v2
export MEETING_STT_BACKEND=wav2vec2
export MEETING_WAV2VEC2_MODEL=kresnik/wav2vec2-large-xlsr-korean
```

검수 모델을 지정하지 않으면 `MEETING_SUMMARY_OLLAMA_MODEL`, `LNPCHAT_LLM_MODEL`, `llama3` 순으로 자동 탐색합니다. 여러 후보를 사용할 경우 `MEETING_SUMMARY_REVIEW_MODEL="llama3.1,phi3"`처럼 쉼표로 나열하세요.

환경 변수 변경 후에는 CLI 혹은 UI를 재시작하여 설정이 반영되었는지 확인합니다.

> 대화형 모드에서 회의 비서를 호출하려면 `/meeting` 명령어를 입력하고 안내에 따라 오디오 파일 경로를 제공하세요.
