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
| `MEETING_STT_BACKEND` | `whisper`, `faster-whisper` 등 STT 엔진 선택 |
| `MEETING_SUMMARY_BACKEND` | `kobart`, `ollama`, `openai` 등 요약 엔진 선택 |
| `MEETING_MASK_PII` | `1`이면 이메일/전화번호를 `[REDACTED_*]` 토큰으로 마스킹 |
| `MEETING_SAVE_TRANSCRIPT` | `1`이면 `transcript.json`을 생성 |

환경 변수 변경 후에는 CLI 혹은 UI를 재시작하여 설정이 반영되었는지 확인합니다.

> 대화형 모드에서 회의 비서를 호출하려면 `/meeting` 명령어를 입력하고 안내에 따라 오디오 파일 경로를 제공하세요.
