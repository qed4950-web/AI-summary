# Meeting Pipeline Models Overview

회의 비서 파이프라인에서 사용하는 모델 유형과 목적을 정리합니다.

| 구성 요소 | 기본 모델 | 설명 | 환경 변수/옵션 |
|-----------|-----------|------|----------------|
| STT (Speech-to-Text) | faster-whisper `small` | 오디오를 텍스트로 변환 | `MEETING_STT_BACKEND`, `MEETING_STT_MODEL`, `MEETING_STT_DEVICE`, `MEETING_STT_COMPUTE` |
| Summariser (KoBART) | `gogamza/kobart-base-v2` | 한국어 문서를 chunk 단위로 요약 | `MEETING_SUMMARY_MODEL` |
| Summariser (BART EN) | `facebook/bart-large-cnn` | 영어 텍스트 요약 | `MEETING_SUMMARY_EN_MODEL` |
| Summariser (Ollama) | 예: `llama3` | 로컬/Ollama 기반 LLM 요약 | `MEETING_SUMMARY_BACKEND=ollama`, `MEETING_SUMMARY_OLLAMA_MODEL`, `MEETING_SUMMARY_OLLAMA_HOST` |
| Summariser (BitNet) | `bitnet/b1.58-instruct` | 경량 LLM 요약 | `MEETING_SUMMARY_BACKEND=bitnet`, `MEETING_SUMMARY_BITNET_MODEL` |
| Summariser (Heuristic) | - | KoBART/LLM 실패 시 키워드 기반 요약 | `MEETING_SUMMARY_BACKEND=heuristic` |

## 요약
- KoBART/BART, BitNet은 **문서 요약 전용** 모델(KoBART, BART, BitNet)입니다.
- Ollama 백엔드는 로컬 LLM을 연결하는 옵션으로, 일반적인 생성형 LLM(예: `llama3`, `mistral`)을 사용할 수 있습니다.
- 백엔드 선택은 `MEETING_SUMMARY_BACKEND` (`kobart`, `ollama`, `bitnet`, `heuristic`) 로 전환합니다.
- STT는 faster-whisper를 기본으로 사용하며, GPU/정밀도 옵션은 환경 변수로 조정할 수 있습니다.
- 현재 파이프라인에는 TTS 단계가 포함되어 있지 않습니다.
