# 회의 파이프라인 모델 요약

회의 비서가 사용하는 주요 모델과 연동 옵션을 정리했습니다.

| 구성 요소 | 기본 모델 | 용도 | 관련 환경 변수 |
|-----------|-----------|------|----------------|
| 음성 → 텍스트(STT, 기본) | Wav2Vec2 `kresnik/wav2vec2-large-xlsr-korean` | 오디오를 문자로 변환 (ko/en 우선) | `MEETING_STT_BACKEND=wav2vec2`, `MEETING_WAV2VEC2_MODEL`, `MEETING_WAV2VEC2_DEVICE`, `MEETING_WAV2VEC2_CHUNK`, `MEETING_WAV2VEC2_STRIDE` |
| 음성 → 텍스트(STT, 대체) | faster-whisper `small` | GPU/CPU 대응 경량 STT | `MEETING_STT_BACKEND=whisper`, `MEETING_STT_MODEL`, `MEETING_STT_DEVICE`, `MEETING_STT_COMPUTE`, `MEETING_STT_MODEL_DIR` |
| 요약기 (KoBART) | `gogamza/kobart-base-v2` | 한국어 요약 | `MEETING_SUMMARY_MODEL` |
| 요약기 (BART EN) | `facebook/bart-large-cnn` | 영어 요약 | `MEETING_SUMMARY_EN_MODEL` |
| 요약기 (Ollama) | 예: `llama3` | 로컬 LLM 기반 요약 | `MEETING_SUMMARY_BACKEND=ollama`, `MEETING_SUMMARY_OLLAMA_MODEL`, `MEETING_SUMMARY_OLLAMA_HOST` |
| 요약기 (BitNet) | `bitnet/b1.58-instruct` | 경량 LLM 요약 | `MEETING_SUMMARY_BACKEND=bitnet`, `MEETING_SUMMARY_BITNET_MODEL` |
| 휴리스틱 요약 | - | 모델 실패 시 키워드 기반 응답 | `MEETING_SUMMARY_BACKEND=heuristic` |
| 요약 감독기 (Supervisor) | 예: `ollama` | 결과 품질 판단/재검수 지시 | `MEETING_SUPERVISOR_BACKEND`, `MEETING_SUPERVISOR_MODEL`, `MEETING_SUPERVISOR_MODE`, `SUMMARY_SUPERVISOR_MODEL`, `DOCUMENT_SUPERVISOR_*` |
| 요약 검수기 (Review) | 예: `ollama` | 1차 요약 검수/보완 | `MEETING_SUMMARY_REVIEW_BACKEND`, `MEETING_SUMMARY_REVIEW_MODEL`, `MEETING_SUMMARY_REVIEW_HOST`, `MEETING_SUMMARY_REVIEW_TIMEOUT` |

## 참고 사항
- KoBART/BART/BitNet은 **요약 전용 모델**이며, Ollama는 원하는 LLM을 지정해 활용할 수 있습니다.
- `MEETING_SUMMARY_BACKEND` 값을 `kobart`, `ollama`, `bitnet`, `heuristic`으로 바꾸어 요약 엔진을 전환합니다.
- STT는 기본적으로 Wav2Vec2(`MEETING_STT_BACKEND=wav2vec2`)를 사용하며, whisper 백엔드는 CPU-only 환경이나 경량 워크로드를 위한 대체 옵션입니다.
- 요약 검수기는 기본적으로 비활성화되어 있으며, `MEETING_SUMMARY_REVIEW_BACKEND`를 지정하면 1차 요약을 LLM으로 보완한 뒤 JSON 결과를 적용합니다.
- `MEETING_SUMMARY_REVIEW_MODEL`을 비워 두거나 `모델A,모델B`처럼 여러 후보를 적어두면, Ollama에 설치된 모델 목록을 확인해 사용 가능한 항목을 자동으로 고릅니다(기본 후보는 `MEETING_SUMMARY_OLLAMA_MODEL` → `LNPCHAT_LLM_MODEL` → `llama3` 순서).
- `MEETING_SUMMARY_REVIEW_MODE`는 `auto`(기본, 결함이 있을 때만 검수), `always`(항상 검수), `manual`/`off`(자동 검수 비활성화)로 동작합니다.
- 감독 LLM을 따로 지정하지 않으면 검수기(`MEETING_SUMMARY_REVIEW_MODEL`) 설정을 재사용하고, 추가 지시가 필요할 때만 `MEETING_SUPERVISOR_*` 변수를 지정하면 됩니다.
- Ollama 기반 모델이 장황하게 응답하면 `MEETING_SUMMARY_OLLAMA_NUM_PREDICT`, `MEETING_SUMMARY_OLLAMA_TEMPERATURE`, `MEETING_SUMMARY_REVIEW_NUM_PREDICT`, `MEETING_SUMMARY_REVIEW_TEMPERATURE`, `SUMMARY_SUPERVISOR_NUM_PREDICT`, `SUMMARY_SUPERVISOR_TEMPERATURE` 등을 조정하세요 (기본 fallback은 `num_predict≈192`, `temperature≈0.08` 수준으로 설정되어 있습니다).
- 현재 파이프라인에는 별도의 음성 합성(TTS) 단계가 포함되어 있지 않습니다.
