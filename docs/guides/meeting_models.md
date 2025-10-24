# 회의 파이프라인 모델 요약

회의 비서가 사용하는 주요 모델과 연동 옵션을 정리했습니다.

| 구성 요소 | 기본 모델 | 용도 | 관련 환경 변수 |
|-----------|-----------|------|----------------|
| 음성 → 텍스트(STT) | faster-whisper `small` | 오디오를 문자로 변환 | `MEETING_STT_BACKEND`, `MEETING_STT_MODEL`, `MEETING_STT_DEVICE`, `MEETING_STT_COMPUTE` |
| 요약기 (KoBART) | `gogamza/kobart-base-v2` | 한국어 요약 | `MEETING_SUMMARY_MODEL` |
| 요약기 (BART EN) | `facebook/bart-large-cnn` | 영어 요약 | `MEETING_SUMMARY_EN_MODEL` |
| 요약기 (Ollama) | 예: `llama3` | 로컬 LLM 기반 요약 | `MEETING_SUMMARY_BACKEND=ollama`, `MEETING_SUMMARY_OLLAMA_MODEL`, `MEETING_SUMMARY_OLLAMA_HOST` |
| 요약기 (BitNet) | `bitnet/b1.58-instruct` | 경량 LLM 요약 | `MEETING_SUMMARY_BACKEND=bitnet`, `MEETING_SUMMARY_BITNET_MODEL` |
| 휴리스틱 요약 | - | 모델 실패 시 키워드 기반 응답 | `MEETING_SUMMARY_BACKEND=heuristic` |

## 참고 사항
- KoBART/BART/BitNet은 **요약 전용 모델**이며, Ollama는 원하는 LLM을 지정해 활용할 수 있습니다.
- `MEETING_SUMMARY_BACKEND` 값을 `kobart`, `ollama`, `bitnet`, `heuristic`으로 바꾸어 요약 엔진을 전환합니다.
- STT는 기본적으로 faster-whisper를 사용하며, GPU 사용 여부와 정밀도는 관련 환경 변수로 조절합니다.
- 현재 파이프라인에는 별도의 음성 합성(TTS) 단계가 포함되어 있지 않습니다.
